#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_carla_reid_dataset.py

Скрипт генерации синтетического датасета для vehicle ReID в CARLA.
"""

import colorsys
import csv
import json
import math
import os
import queue as py_queue
import random
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import carla


@dataclass
class Identity:
    """Описание одной уникальной идентичности автомобиля."""
    identity_id: int
    split: str  # train / test
    blueprint: str
    base_color: Tuple[int, int, int]
    appearance_seed: int
    target_train: int = 0
    target_query: int = 0
    target_gallery: int = 0


@dataclass
class CameraSpec:
    """Спецификация камеры относительно автомобиля."""
    camera_id: str
    transform: carla.Transform
    fov: float


@dataclass
class CameraRuntime:
    """Рабочая камера с сенсором и очередью изображений."""
    camera_id: str
    actor: carla.Actor
    queue: Any
    intrinsic: np.ndarray

def log(message: str) -> None:
    """Печать статуса с префиксом."""
    print(f"[CARLA-ReID] {message}", flush=True)


def safe_destroy(actor: Optional[carla.Actor]) -> None:
    """Безопасное удаление актёра."""
    if actor is None:
        return
    try:
        actor.destroy()
    except Exception as exc:
        log(f"Не удалось удалить актёра: {exc}")


def delete_records_files(records: List[Dict[str, Any]], cfg: Any) -> None:
    """Удалить файлы изображений для отброшенных записей."""
    for rec in records:
        path = os.path.join(cfg.output_dir, "images", rec["image_id"])
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass



# Подключение и настройка CARLA
def connect_client(host: str, port: int, timeout: float) -> carla.Client:
    """Подключиться к CARLA серверу с понятной ошибкой, если сервер не запущен."""
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        server_version = client.get_server_version()
        log(f"CARLA server version: {server_version}")
        return client
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось подключиться к CARLA на {host}:{port}. "
            f"Проверьте, запущен ли сервер. Ошибка: {exc}"
        )


def setup_world(client: carla.Client, cfg: Any) -> Tuple[carla.World, Any, Any]:
    """
    Настроить мир: при необходимости загрузить карту, включить синхронный режим.
    Возвращает мир, исходные настройки и исходную погоду для восстановления.
    """
    world = client.get_world()
    original_settings = world.get_settings()
    original_weather = world.get_weather()

    current_map_name = os.path.basename(world.get_map().name)
    if current_map_name != cfg.town:
        try:
            log(f"Загрузка карты {cfg.town} вместо текущей {current_map_name}...")
            client.load_world(cfg.town)
            time.sleep(cfg.map_load_wait)
            world = client.get_world()
            original_settings = world.get_settings()
            original_weather = world.get_weather()
        except Exception as exc:
            log(f"Не удалось загрузить карту {cfg.town}, используем текущую: {exc}")

    try:
        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=cfg.fixed_delta_seconds,
            no_rendering_mode=False,
        )
    except TypeError:
        # Для старых версий, где нет каких-то параметров.
        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=cfg.fixed_delta_seconds,
        )

    world.apply_settings(settings)
    world.tick()
    return world, original_settings, original_weather


def setup_traffic_manager(client: carla.Client, cfg: Any) -> Optional[Any]:
    """Настроить Traffic Manager в синхронном режиме, если автопилот будет использоваться."""
    try:
        tm = client.get_trafficmanager(cfg.traffic_manager_port)
        tm.set_synchronous_mode(True)

        try:
            tm.set_random_device_seed(cfg.seed)
        except Exception:
            pass

        # Необязательные настройки, чтобы машины чаще двигались.
        try:
            tm.global_percentage_distance_to_leading_vehicle(25.0)
        except Exception:
            pass
        try:
            tm.ignore_lights_percentage(65.0)
        except Exception:
            pass
        try:
            tm.ignore_signs_percentage(65.0)
        except Exception:
            pass
        try:
            tm.set_hybrid_physics_mode(True)
        except Exception:
            pass

        return tm
    except Exception as exc:
        log(f"Traffic Manager недоступен, автопилот может работать ограниченно: {exc}")
        return None


def get_available_vehicle_blueprints(world: carla.World, cfg: Any) -> List[str]:
    """
    Получить доступные четырёхколёсные автомобили.
    Сначала ищем предпочтительные из конфига, затем любой доступный транспорт.
    """
    lib = world.get_blueprint_library()
    vehicles = lib.filter("vehicle.*")
    preferred = set(cfg.blueprint_filter)
    result: List[str] = []

    # Сначала предпочтительные модели с 4 колёсами.
    for bp in vehicles:
        try:
            wheels = 4
            if bp.has_attribute("number_of_wheels"):
                wheels = int(bp.get_attribute("number_of_wheels"))
            if wheels < 4:
                continue
            if bp.id in preferred:
                result.append(bp.id)
        except Exception:
            continue

    # Если предпочтительных нет, берём доступные четырёхколёсные.
    if not result:
        for bp in vehicles:
            try:
                wheels = 4
                if bp.has_attribute("number_of_wheels"):
                    wheels = int(bp.get_attribute("number_of_wheels"))
                if wheels >= 4:
                    result.append(bp.id)
            except Exception:
                continue

    # Последний fallback: вообще все транспортные blueprint'ы.
    if not result:
        result = [bp.id for bp in vehicles]

    result = sorted(set(result))
    if not result:
        raise RuntimeError("В текущем мире нет доступных транспортных средств для генерации датасета.")
    return result



# Погода

def make_weather(params: Dict[str, float]) -> carla.WeatherParameters:
    """Создать погоду с безопасным fallback."""
    try:
        return carla.WeatherParameters(**params)
    except Exception:
        try:
            return carla.WeatherParameters.ClearNoon
        except Exception:
            return carla.WeatherParameters()


def get_weather_presets() -> List[Tuple[str, carla.WeatherParameters]]:
    """
    Набор погодных пресетов: день, облачность, дождь, мокрая дорога,
    закат, туман, вечер. Ночь исключена, чтобы избежать слишком тёмных кадров.
    """
    raw = [
        (
            "clear_noon",
            dict(
                cloudiness=5.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=10.0,
                sun_azimuth_angle=45.0,
                sun_altitude_angle=70.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=10.0,
            ),
        ),
        (
            "cloudy_noon",
            dict(
                cloudiness=65.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=15.0,
                sun_azimuth_angle=60.0,
                sun_altitude_angle=62.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=15.0,
            ),
        ),
        (
            "wet_noon",
            dict(
                cloudiness=45.0,
                precipitation=0.0,
                precipitation_deposits=80.0,
                wind_intensity=12.0,
                sun_azimuth_angle=40.0,
                sun_altitude_angle=58.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=90.0,
            ),
        ),
        (
            "rain_noon",
            dict(
                cloudiness=85.0,
                precipitation=55.0,
                precipitation_deposits=90.0,
                wind_intensity=25.0,
                sun_azimuth_angle=50.0,
                sun_altitude_angle=52.0,
                fog_density=2.0,
                fog_distance=20.0,
                wetness=100.0,
            ),
        ),
        (
            "clear_sunset",
            dict(
                cloudiness=10.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=8.0,
                sun_azimuth_angle=270.0,
                sun_altitude_angle=14.0,
                fog_density=0.0,
                fog_distance=0.0,
                wetness=10.0,
            ),
        ),
        (
            "cloudy_sunset",
            dict(
                cloudiness=75.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=15.0,
                sun_azimuth_angle=260.0,
                sun_altitude_angle=11.0,
                fog_density=1.0,
                fog_distance=25.0,
                wetness=25.0,
            ),
        ),
        (
            "wet_sunset",
            dict(
                cloudiness=65.0,
                precipitation=0.0,
                precipitation_deposits=75.0,
                wind_intensity=18.0,
                sun_azimuth_angle=255.0,
                sun_altitude_angle=9.0,
                fog_density=1.0,
                fog_distance=25.0,
                wetness=85.0,
            ),
        ),
        (
            "soft_rain_sunset",
            dict(
                cloudiness=80.0,
                precipitation=35.0,
                precipitation_deposits=85.0,
                wind_intensity=22.0,
                sun_azimuth_angle=250.0,
                sun_altitude_angle=8.0,
                fog_density=3.0,
                fog_distance=18.0,
                wetness=100.0,
            ),
        ),
        (
            "foggy_dawn",
            dict(
                cloudiness=40.0,
                precipitation=0.0,
                precipitation_deposits=20.0,
                wind_intensity=5.0,
                sun_azimuth_angle=90.0,
                sun_altitude_angle=6.0,
                fog_density=18.0,
                fog_distance=40.0,
                wetness=30.0,
            ),
        ),
        (
            "dusk",
            dict(
                cloudiness=35.0,
                precipitation=0.0,
                precipitation_deposits=10.0,
                wind_intensity=8.0,
                sun_azimuth_angle=265.0,
                sun_altitude_angle=3.0,
                fog_density=1.0,
                fog_distance=30.0,
                wetness=20.0,
            ),
        ),
    ]

    presets: List[Tuple[str, carla.WeatherParameters]] = []
    for name, params in raw:
        presets.append((name, make_weather(params)))
    return presets


def choose_weathers(rng: random.Random, count: int) -> List[carla.WeatherParameters]:
    """Выбрать несколько различных погодных условий."""
    presets = [w for _, w in get_weather_presets()]
    if not presets:
        return [make_weather({})]
    count = max(1, min(int(count), len(presets)))
    return rng.sample(presets, count)


def set_weather(world: carla.World, weather: carla.WeatherParameters, ticks: int = 2) -> None:
    """Применить погоду и сделать несколько тиков, чтобы изменения вступили в силу."""
    world.set_weather(weather)
    for _ in range(max(0, ticks)):
        world.tick()


# Генерация идентичностей

def generate_base_color(rng: random.Random) -> Tuple[int, int, int]:
    """Сгенерировать реалистичный базовый цвет автомобиля."""
    if rng.random() < 0.18:
        # Серые/белые/чёрные оттенки.
        v = rng.uniform(0.12, 0.96)
        c = int(v * 255)
        return c, c, c

    h = rng.random()
    s = rng.uniform(0.20, 0.90)
    v = rng.uniform(0.25, 0.98)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def generate_identity_specs(cfg: Any, blueprints: List[str]) -> Tuple[List[Identity], List[Identity]]:
    """Создать идентичности для обучения и теста."""
    rng = random.Random(cfg.seed)
    train_identities: List[Identity] = []
    test_identities: List[Identity] = []

    for i in range(cfg.num_train_identities):
        identity_id = cfg.train_id_start + i
        blueprint = rng.choice(blueprints)
        base_color = generate_base_color(rng)
        appearance_seed = (cfg.seed * 1000003 + identity_id * 9176 + 12345) & 0xFFFFFFFF

        ident = Identity(
            identity_id=identity_id,
            split="train",
            blueprint=blueprint,
            base_color=base_color,
            appearance_seed=appearance_seed,
            target_train=int(rng.randint(cfg.min_train_images_per_identity, cfg.max_train_images_per_identity + 1)),
        )
        train_identities.append(ident)

    for j in range(cfg.num_test_identities):
        identity_id = cfg.test_id_start + j
        blueprint = rng.choice(blueprints)
        base_color = generate_base_color(rng)
        appearance_seed = (cfg.seed * 2000003 + identity_id * 7331 + 54321) & 0xFFFFFFFF

        ident = Identity(
            identity_id=identity_id,
            split="test",
            blueprint=blueprint,
            base_color=base_color,
            appearance_seed=appearance_seed,
            target_query=cfg.num_test_query_per_identity,
            target_gallery=cfg.num_test_gallery_per_identity,
        )
        test_identities.append(ident)

    return train_identities, test_identities



# Камеры

def build_camera_specs_for_vehicle(vehicle: carla.Actor, cfg: Any) -> List[CameraSpec]:
    """
    Создать спецификации камер вокруг автомобиля.

    Убран вид сверху. Камеры расположены более реалистично:
    - сзади на уровне примерно 2 метра;
    - спереди на уровне примерно 1.7-1.8 метра;
    - сбоку на уровне примерно 1.7 метра;
    - дальние камеры на уровне 2.2-2.6 метра.
    """
    extent = vehicle.bounding_box.extent
    local_target = vehicle.bounding_box.location

    length = max(3.0, extent.x * 2.0)
    width = max(1.5, extent.y * 2.0)

    rear_dist = max(6.5, length * 1.2 + 2.5)
    front_dist = max(6.5, length * 1.2 + 2.5)
    side_dist = max(3.5, width * 1.4 + 2.0)

    far_rear_dist = max(12.0, length * 2.2 + 5.0)
    far_front_dist = max(12.0, length * 2.2 + 5.0)

    def make_spec(
        camera_id: str,
        px: float,
        py: float,
        pz: float,
        fov: float,
    ) -> CameraSpec:
        """
        Камера смотрит примерно на центр бокса автомобиля.
        Углы рассчитываются по локальному смещению камеры и целевой точке.
        """
        dx = local_target.x - px
        dy = local_target.y - py
        dz = local_target.z - pz

        yaw = math.degrees(math.atan2(dy, dx))
        horizontal = math.sqrt(dx * dx + dy * dy)

        if horizontal > 1e-3:
            pitch = math.degrees(math.atan2(dz, horizontal))
        else:
            pitch = -8.0

        # Ограничиваем наклон, чтобы не было чрезмерного взгляда вниз/вверх.
        pitch = max(-16.0, min(-5.0, pitch))

        transform = carla.Transform(
            carla.Location(x=px, y=py, z=pz),
            carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
        )

        return CameraSpec(camera_id=camera_id, transform=transform, fov=fov)

    specs = [
        # Задняя камера, как сопровождение/дорожная камера позади машины.
        make_spec("cam00_rear", -rear_dist, 0.0, 2.1, 90.0),

        # Фронтальная камера спереди, смотрит назад на автомобиль.
        make_spec("cam01_front", front_dist, 0.0, 1.8, 90.0),

        # Левая боковая камера.
        make_spec("cam02_left", -0.8, -side_dist, 1.7, 100.0),

        # Правая боковая камера.
        make_spec("cam03_right", -0.8, side_dist, 1.7, 100.0),

        # Дальняя задняя камера, более узкий угол, как телеобъектив/камера наблюдения.
        make_spec("cam04_rear_far", -far_rear_dist, 1.6, 2.6, 55.0),

        # Дальняя фронтальная камера, более узкий угол.
        make_spec("cam05_front_far", far_front_dist, -1.6, 2.2, 55.0),
    ]

    return specs


def make_intrinsic(width: int, height: int, fov: float) -> np.ndarray:
    """Матрица внутриненных параметров камеры. Предполагаем горизонтальный FOV."""
    f = width / (2.0 * math.tan(math.radians(fov) * 0.5))
    return np.array(
        [
            [f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def set_optional_attribute(bp: carla.ActorBlueprint, name: str, value: str) -> None:
    """Безопасно выставить атрибут камеры, если он доступен."""
    try:
        if bp.has_attribute(name):
            bp.set_attribute(name, value)
    except Exception:
        pass


def create_camera_rig(
    world: carla.World,
    vehicle: carla.Actor,
    identity: Identity,
    cfg: Any,
) -> List[CameraRuntime]:
    """Создать набор RGB-камер, прикреплённых к автомобилю."""
    specs = build_camera_specs_for_vehicle(vehicle, cfg)
    cameras: List[CameraRuntime] = []
    bp_lib = world.get_blueprint_library()

    # Используем Rigid, чтобы камеры гарантированно сохраняли заданный ракурс.
    attachment = getattr(carla.AttachmentType, "Rigid", None)
    if attachment is None:
        attachment = getattr(carla.AttachmentType, "SpringArm", None)

    for spec in specs:
        try:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(cfg.image_width))
            bp.set_attribute("image_size_y", str(cfg.image_height))
            bp.set_attribute("fov", str(spec.fov))
            bp.set_attribute("sensor_tick", "0.0")

            # Аккуратные настройки изображения.
            set_optional_attribute(bp, "gamma", "2.2")
            set_optional_attribute(bp, "motion_blur_intensity", "0.0")
            set_optional_attribute(bp, "lens_circle_falloff", "0.0")
            set_optional_attribute(bp, "chromatic_aberration_intensity", "0.0")
            set_optional_attribute(bp, "chromatic_aberration_offset", "0.0")
            set_optional_attribute(bp, "exposure_mode", "auto")

            if bp.has_attribute("role_name"):
                try:
                    bp.set_attribute("role_name", f"reid_cam_{identity.identity_id}_{spec.camera_id}")
                except Exception:
                    pass

            if attachment is not None:
                sensor = world.spawn_actor(bp, spec.transform, attach_to=vehicle, attachment_type=attachment)
            else:
                sensor = world.spawn_actor(bp, spec.transform, attach_to=vehicle)

            q = py_queue.Queue()
            sensor.listen(lambda image, qq=q: qq.put(image))

            cameras.append(
                CameraRuntime(
                    camera_id=spec.camera_id,
                    actor=sensor,
                    queue=q,
                    intrinsic=make_intrinsic(cfg.image_width, cfg.image_height, spec.fov),
                )
            )
        except Exception as exc:
            log(f"Не удалось создать камеру {spec.camera_id} для identity {identity.identity_id}: {exc}")

    return cameras


def destroy_camera_rig(cameras: List[CameraRuntime]) -> None:
    """Остановить и удалить все камеры."""
    for cam in cameras:
        try:
            cam.actor.stop()
        except Exception:
            pass
        safe_destroy(cam.actor)


def drain_queues(cameras: List[CameraRuntime]) -> None:
    """Очистить очереди кадров перед следующим тиком."""
    for cam in cameras:
        while True:
            try:
                cam.queue.get_nowait()
            except py_queue.Empty:
                break


# ============================================================
# Транспорт
# ============================================================

def spawn_vehicle_for_identity(
    world: carla.World,
    identity: Identity,
    spawn_points: List[carla.Transform],
    rng: random.Random,
    cfg: Any,
) -> Optional[carla.Actor]:
    """
    Заспавнить автомобиль идентичности, применить цвет и доступные атрибуты.
    Если точный цвет не ставится, используется рекомендуемое значение.
    """
    lib = world.get_blueprint_library()
    try:
        bp = lib.find(identity.blueprint)
    except Exception:
        bp = None

    if bp is None:
        log(f"Blueprint {identity.blueprint} не найден для identity {identity.identity_id}.")
        return None

    # Роль для последующей очистки.
    if bp.has_attribute("role_name"):
        try:
            bp.set_attribute("role_name", f"reid_vehicle_{identity.identity_id}")
        except Exception:
            pass

    # Базовый цвет.
    if bp.has_attribute("color"):
        color_attr = bp.get_attribute("color")
        r, g, b = identity.base_color
        color_set = False

        # Иногда используем нативные рекомендуемые цвета, они обычно выглядят корректнее.
        try:
            recommended = color_attr.recommended_values
            if recommended and rng.random() < 0.5:
                bp.set_attribute("color", rng.choice(recommended))
                color_set = True
        except Exception:
            color_set = False

        if not color_set:
            try:
                bp.set_attribute("color", f"{r},{g},{b}")
                color_set = True
            except Exception:
                try:
                    recommended = color_attr.recommended_values
                    if recommended:
                        bp.set_attribute("color", rng.choice(recommended))
                        color_set = True
                except Exception:
                    color_set = False

    # Если есть скины/варианты материалов, используем их детерминированно.
    if bp.has_attribute("skin"):
        try:
            skin_attr = bp.get_attribute("skin")
            values = skin_attr.recommended_values
            if values:
                skin_rng = random.Random(identity.appearance_seed ^ 0x5F3A)
                if skin_rng.random() < 0.5:
                    bp.set_attribute("skin", skin_rng.choice(values))
        except Exception:
            pass

    for _ in range(cfg.spawn_attempts_per_vehicle):
        spawn_point = rng.choice(spawn_points)
        try:
            vehicle = world.try_spawn_actor(bp, spawn_point)
        except Exception:
            vehicle = None

        if vehicle is None:
            continue

        # Автопилот.
        try:
            vehicle.set_autopilot(True, cfg.traffic_manager_port)
        except Exception:
            try:
                vehicle.set_autopilot(True)
            except Exception:
                pass

        # Свет автомобиля для тёмных условий.
        try:
            vehicle.set_light_state(carla.VehicleLightState.All)
        except Exception:
            pass

        return vehicle

    return None


# ============================================================
# Проекция 3D -> 2D и bounding boxes
# ============================================================

def project_point(
    location: carla.Location,
    camera_transform: carla.Transform,
    intrinsic: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Спроецировать мировую точку в изображение камеры."""
    world_2_camera = np.array(camera_transform.get_inverse_matrix(), dtype=np.float64)
    point_world = np.array([location.x, location.y, location.z, 1.0], dtype=np.float64)
    point_camera = world_2_camera @ point_world

    # Вперёд по оси X в системе камеры.
    if point_camera[0] <= 0.15:
        return None

    # Переход к координатам изображения:
    # x_image ~ y_camera, y_image ~ -z_camera.
    point_image = intrinsic @ np.array(
        [point_camera[1], -point_camera[2], point_camera[0]],
        dtype=np.float64,
    )

    u = point_image[0] / point_camera[0]
    v = point_image[1] / point_camera[0]
    return float(u), float(v)


def get_vehicle_world_corners(vehicle: carla.Actor) -> List[carla.Location]:
    """Получить 8 мировых углов 3D-бокса автомобиля."""
    bb = vehicle.bounding_box
    extent = bb.extent
    local_center = bb.location
    transform_matrix = np.array(vehicle.get_transform().get_matrix(), dtype=np.float64)

    corners: List[carla.Location] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local_point = np.array(
                    [
                        local_center.x + sx * extent.x,
                        local_center.y + sy * extent.y,
                        local_center.z + sz * extent.z,
                        1.0,
                    ],
                    dtype=np.float64,
                )
                world_point = transform_matrix @ local_point
                corners.append(
                    carla.Location(
                        x=float(world_point[0]),
                        y=float(world_point[1]),
                        z=float(world_point[2]),
                    )
                )
    return corners


def get_actor_center_world(actor: carla.Actor) -> carla.Location:
    """Мировой центр бокса актёра."""
    transform_matrix = np.array(actor.get_transform().get_matrix(), dtype=np.float64)
    local_center = actor.bounding_box.location
    p = np.array([local_center.x, local_center.y, local_center.z, 1.0], dtype=np.float64)
    world_p = transform_matrix @ p
    return carla.Location(x=float(world_p[0]), y=float(world_p[1]), z=float(world_p[2]))


def compute_vehicle_bbox_2d(
    vehicle: carla.Actor,
    camera_transform: carla.Transform,
    intrinsic: np.ndarray,
    cfg: Any,
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Спроецировать 3D-бокс автомобиля в 2D и вернуть (x, y, w, h, visible_ratio).
    Отбрасываем слишком маленькие, сильно обрезанные и слишком большие боксы.
    """
    corners = get_vehicle_world_corners(vehicle)
    projected: List[Tuple[float, float]] = []

    for corner in corners:
        p = project_point(corner, camera_transform, intrinsic)
        if p is None:
            return None
        projected.append(p)

    pts = np.array(projected, dtype=np.float64)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)

    if x2 <= x1 or y2 <= y1:
        return None

    original_area = max(float((x2 - x1) * (y2 - y1)), 1e-6)

    cx1 = max(0.0, float(x1))
    cy1 = max(0.0, float(y1))
    cx2 = min(float(cfg.image_width), float(x2))
    cy2 = min(float(cfg.image_height), float(y2))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    visible_area = float((cx2 - cx1) * (cy2 - cy1))
    visible_ratio = visible_area / original_area

    if visible_ratio < cfg.min_visible_fraction:
        return None

    x = int(round(cx1))
    y = int(round(cy1))
    w = int(round(cx2 - cx1))
    h = int(round(cy2 - cy1))

    x = max(0, min(x, cfg.image_width - 1))
    y = max(0, min(y, cfg.image_height - 1))
    w = min(w, cfg.image_width - x)
    h = min(h, cfg.image_height - y)

    if w <= 0 or h <= 0:
        return None

    area = w * h
    if area < cfg.min_bbox_area:
        return None

    if area > cfg.max_bbox_area_fraction * cfg.image_width * cfg.image_height:
        return None

    return x, y, w, h, float(visible_ratio)


def is_visible_with_fallback(
    world: carla.World,
    camera_transform: carla.Transform,
    vehicle: carla.Actor,
    cfg: Any,
) -> bool:
    """
    Проверка видимости.
    Здесь используется программный fallback: raycast до центра автомобиля, если API доступен.
    Если функция недоступна, считаем объект видимым после проверки бокса.
    """
    if not cfg.use_visibility_raycast_fallback:
        return True

    try:
        origin = camera_transform.location
        target = get_actor_center_world(vehicle)
        target_distance = origin.distance(target)

        cast_func = getattr(world, "cast_ray", None)
        if cast_func is None:
            cast_func = getattr(world, "ray_cast", None)
        if cast_func is None:
            return True

        hits = cast_func(origin, target)
        if hits is None:
            return True

        if not isinstance(hits, (list, tuple)):
            hits = [hits]

        if len(hits) == 0:
            return True

        closest = min(hits, key=lambda loc: origin.distance(loc))
        closest_distance = origin.distance(closest)

        # Если луч ударился существенно раньше центра машины, вероятно, объект перекрыт.
        if closest_distance < target_distance - 1.5:
            return False

        return True
    except Exception:
        return True


# ============================================================
# Сохранение изображений
# ============================================================

def carla_image_to_bgr(image: carla.Image) -> Optional[np.ndarray]:
    """Конвертировать CARLA image в BGR для OpenCV."""
    try:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = arr.reshape((image.height, image.width, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as exc:
        log(f"Не удалось сконвертировать изображение: {exc}")
        return None


def save_frame(
    image: carla.Image,
    bbox: Tuple[int, int, int, int, float],
    identity: Identity,
    cfg: Any,
    image_id: str,
) -> bool:
    """
    Сохранить кадр без процедурных наклеек/царапин/повреждений.
    Используется только исходное изображение из CARLA.
    """
    img = carla_image_to_bgr(image)
    if img is None:
        return False

    path = os.path.join(cfg.output_dir, "images", image_id)
    try:
        return bool(cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]))
    except Exception as exc:
        log(f"Не удалось сохранить изображение {path}: {exc}")
        return False


# ============================================================
# Захват изображений
# ============================================================

def capture_phase(
    world: carla.World,
    vehicle: carla.Actor,
    cameras: List[CameraRuntime],
    identity: Identity,
    split_prefix: str,
    needed: int,
    allowed_camera_ids: List[str],
    cfg: Any,
    counters: Dict[str, int],
    seq_start: int,
    cam_counts: Dict[str, int],
    last_frame: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int], Dict[str, int]]:
    """
    Захватить нужное количество изображений для одной фазы.
    Возвращает записи, следующий seq, счётчики камер и последние кадры.
    """
    records: List[Dict[str, Any]] = []
    seq = seq_start
    ticks = 0
    failed_ticks = 0
    allowed = set(allowed_camera_ids)

    while len(records) < needed and ticks < cfg.max_ticks_per_phase:
        drain_queues(cameras)

        try:
            world.tick()
        except Exception as exc:
            log(f"Ошибка тика мира: {exc}")
            counters["discarded"] += 1
            break

        frame_id = world.get_snapshot().frame
        frames: Dict[str, Any] = {}
        missing = False

        for cam in cameras:
            try:
                img = cam.queue.get(timeout=cfg.sensor_queue_timeout)
                frames[cam.camera_id] = img
            except py_queue.Empty:
                missing = True
                break

        if missing:
            counters["discarded"] += 1
            failed_ticks += 1
            if failed_ticks > cfg.max_failed_ticks:
                break
            continue

        ticks += 1
        saved_this_tick = 0

        for cam in cameras:
            cid = cam.camera_id
            if cid not in allowed:
                continue

            if cam_counts.get(cid, 0) >= cfg.max_images_per_camera_per_identity:
                continue

            img = frames[cid]
            img_frame = getattr(img, "frame", frame_id)

            if img_frame - last_frame.get(cid, -1_000_000) < cfg.capture_cooldown_frames:
                continue

            cam_transform = getattr(img, "transform", None)
            if cam_transform is None:
                cam_transform = cam.actor.get_transform()

            bbox = compute_vehicle_bbox_2d(vehicle, cam_transform, cam.intrinsic, cfg)
            if bbox is None:
                counters["discarded"] += 1
                continue

            if not is_visible_with_fallback(world, cam_transform, vehicle, cfg):
                counters["discarded"] += 1
                continue

            image_id = f"{split_prefix}_{identity.identity_id:06d}_{seq:04d}.jpg"

            if save_frame(img, bbox, identity, cfg, image_id):
                records.append(
                    {
                        "image_id": image_id,
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2],
                        "h": bbox[3],
                        "vehicle_id": identity.identity_id,
                    }
                )
                seq += 1
                cam_counts[cid] = cam_counts.get(cid, 0) + 1
                last_frame[cid] = img_frame
                saved_this_tick += 1

                if len(records) >= needed:
                    break

        if saved_this_tick == 0:
            failed_ticks += 1
        else:
            failed_ticks = 0

        if failed_ticks > cfg.max_failed_ticks:
            break

    return records, seq, cam_counts, last_frame


def process_train_identity(
    world: carla.World,
    identity: Identity,
    spawn_points: List[carla.Transform],
    cfg: Any,
    counters: Dict[str, int],
    master_rng: random.Random,
) -> List[Dict[str, Any]]:
    """Сгенерировать обучающие изображения для одной идентичности."""
    min_needed = max(2, cfg.min_train_images_per_identity)

    for attempt in range(cfg.max_spawn_attempts):
        rng = random.Random(master_rng.randint(0, 2**31 - 1))
        vehicle = spawn_vehicle_for_identity(world, identity, spawn_points, rng, cfg)
        if vehicle is None:
            continue

        cameras = create_camera_rig(world, vehicle, identity, cfg)
        if not cameras:
            safe_destroy(vehicle)
            continue

        records: List[Dict[str, Any]] = []
        result: List[Dict[str, Any]] = []

        try:
            # Прогрев камер.
            world.tick()
            time.sleep(0.15)
            world.tick()
            drain_queues(cameras)

            camera_ids = [c.camera_id for c in cameras]
            cam_counts: Dict[str, int] = {}
            last_frame: Dict[str, int] = {}
            seq = 0

            weathers = choose_weathers(rng, cfg.weather_variants_per_identity)
            if not weathers:
                weathers = [make_weather({})]

            chunk_index = 0
            max_chunks = max(
                4,
                (identity.target_train // max(1, cfg.images_per_weather_chunk))
                + cfg.weather_variants_per_identity
                + 3,
            )

            while len(records) < identity.target_train and chunk_index < max_chunks:
                weather = weathers[chunk_index % len(weathers)]
                set_weather(world, weather, ticks=2)

                needed = min(cfg.images_per_weather_chunk, identity.target_train - len(records))
                new_records, seq, cam_counts, last_frame = capture_phase(
                    world=world,
                    vehicle=vehicle,
                    cameras=cameras,
                    identity=identity,
                    split_prefix="train",
                    needed=needed,
                    allowed_camera_ids=camera_ids,
                    cfg=cfg,
                    counters=counters,
                    seq_start=seq,
                    cam_counts=cam_counts,
                    last_frame=last_frame,
                )

                records.extend(new_records)
                chunk_index += 1

                # Если ничего не поймали, немного продвинем сцену.
                if not new_records:
                    for _ in range(5):
                        world.tick()

            if len(records) >= min_needed:
                result = records
            else:
                delete_records_files(records, cfg)

        except Exception as exc:
            log(f"Ошибка при генерации train identity {identity.identity_id}: {exc}")
            delete_records_files(records, cfg)
        finally:
            destroy_camera_rig(cameras)
            safe_destroy(vehicle)

        if result:
            return result

    return []


def process_test_identity(
    world: carla.World,
    identity: Identity,
    spawn_points: List[carla.Transform],
    cfg: Any,
    counters: Dict[str, int],
    master_rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Сгенерировать query и gallery изображения для одной тестовой идентичности."""
    for attempt in range(cfg.max_spawn_attempts):
        rng = random.Random(master_rng.randint(0, 2**31 - 1))
        vehicle = spawn_vehicle_for_identity(world, identity, spawn_points, rng, cfg)
        if vehicle is None:
            continue

        cameras = create_camera_rig(world, vehicle, identity, cfg)
        if not cameras:
            safe_destroy(vehicle)
            continue

        query_records: List[Dict[str, Any]] = []
        gallery_records: List[Dict[str, Any]] = []
        query_result: List[Dict[str, Any]] = []
        gallery_result: List[Dict[str, Any]] = []

        try:
            # Прогрев камер.
            world.tick()
            time.sleep(0.15)
            world.tick()
            drain_queues(cameras)

            camera_ids = [c.camera_id for c in cameras]

            if len(camera_ids) >= 2:
                query_allowed = rng.sample(camera_ids, k=min(3, len(camera_ids)))
                gallery_allowed = [cid for cid in camera_ids if cid not in query_allowed]
                if not gallery_allowed:
                    gallery_allowed = camera_ids
            else:
                query_allowed = camera_ids
                gallery_allowed = camera_ids

            weathers = choose_weathers(rng, count=4)
            if len(weathers) < 2:
                weathers = [make_weather({}), make_weather({"cloudiness": 50.0})]

            query_weather = weathers[0]
            gallery_weather = weathers[1]

            # Query: несколько попыток.
            for retry in range(cfg.max_capture_retries):
                if retry == 0:
                    set_weather(world, query_weather, ticks=2)
                    allowed = query_allowed
                else:
                    set_weather(world, choose_weathers(rng, 1)[0], ticks=2)
                    allowed = camera_ids

                query_records, _, _, _ = capture_phase(
                    world=world,
                    vehicle=vehicle,
                    cameras=cameras,
                    identity=identity,
                    split_prefix="test_query",
                    needed=identity.target_query,
                    allowed_camera_ids=allowed,
                    cfg=cfg,
                    counters=counters,
                    seq_start=0,
                    cam_counts={},
                    last_frame={},
                )

                if len(query_records) >= identity.target_query:
                    break

                delete_records_files(query_records, cfg)
                query_records = []

            if len(query_records) < identity.target_query:
                continue

            # Gallery: несколько попыток.
            for retry in range(cfg.max_capture_retries):
                if retry == 0:
                    set_weather(world, gallery_weather, ticks=2)
                    allowed = gallery_allowed
                else:
                    set_weather(world, choose_weathers(rng, 1)[0], ticks=2)
                    allowed = camera_ids

                gallery_records, _, _, _ = capture_phase(
                    world=world,
                    vehicle=vehicle,
                    cameras=cameras,
                    identity=identity,
                    split_prefix="test_gallery",
                    needed=identity.target_gallery,
                    allowed_camera_ids=allowed,
                    cfg=cfg,
                    counters=counters,
                    seq_start=0,
                    cam_counts={},
                    last_frame={},
                )

                if len(gallery_records) >= identity.target_gallery:
                    break

                delete_records_files(gallery_records, cfg)
                gallery_records = []

            if len(gallery_records) >= identity.target_gallery:
                query_result = query_records
                gallery_result = gallery_records
            else:
                delete_records_files(query_records + gallery_records, cfg)

        except Exception as exc:
            log(f"Ошибка при генерации test identity {identity.identity_id}: {exc}")
            delete_records_files(query_records + gallery_records, cfg)
        finally:
            destroy_camera_rig(cameras)
            safe_destroy(vehicle)

        if query_result and gallery_result:
            return query_result, gallery_result

    return [], []


# ============================================================
# Запись CSV и meta
# ============================================================

def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """Записать CSV с нужными колонками."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(clean_row)


def write_datasets(
    cfg: Any,
    train_records: List[Dict[str, Any]],
    query_records: List[Dict[str, Any]],
    gallery_records: List[Dict[str, Any]],
    ground_truth: Dict[str, int],
    successful_identities: List[Identity],
    extra_info: Dict[str, Any],
) -> None:
    """Сохранить CSV и метафайлы."""
    train_path = os.path.join(cfg.output_dir, "train.csv")
    query_path = os.path.join(cfg.output_dir, "test_query.csv")
    gallery_path = os.path.join(cfg.output_dir, "test_gallery.csv")

    write_csv(train_path, ["image_id", "x", "y", "w", "h", "vehicle_id"], train_records)
    write_csv(query_path, ["image_id", "x", "y", "w", "h"], query_records)
    write_csv(gallery_path, ["image_id", "x", "y", "w", "h"], gallery_records)

    meta_dir = os.path.join(cfg.output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    try:
        cfg_dict = asdict(cfg)
    except Exception:
        cfg_dict = vars(cfg).copy()

    dataset_info = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg_dict,
        "counts": {
            "train_images": len(train_records),
            "test_query_images": len(query_records),
            "test_gallery_images": len(gallery_records),
            "successful_identities": len(successful_identities),
        },
        "extra": extra_info,
    }

    with open(os.path.join(meta_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    appearances = []
    for ident in successful_identities:
        appearances.append(
            {
                "identity_id": ident.identity_id,
                "split": ident.split,
                "blueprint": ident.blueprint,
                "base_color": list(ident.base_color),
                "appearance_seed": ident.appearance_seed,
                "target_train": ident.target_train,
                "target_query": ident.target_query,
                "target_gallery": ident.target_gallery,
            }
        )

    with open(os.path.join(meta_dir, "identities_appearance.json"), "w", encoding="utf-8") as f:
        json.dump(appearances, f, ensure_ascii=False, indent=2)

    with open(os.path.join(meta_dir, "test_ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)


# ============================================================
# Валидация датасета
# ============================================================

def validate_dataset(cfg: Any) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Полная проверка сгенерированного датасета."""
    errors: List[str] = []
    images_dir = os.path.join(cfg.output_dir, "images")

    train_path = os.path.join(cfg.output_dir, "train.csv")
    query_path = os.path.join(cfg.output_dir, "test_query.csv")
    gallery_path = os.path.join(cfg.output_dir, "test_gallery.csv")
    gt_path = os.path.join(cfg.output_dir, "meta", "test_ground_truth.json")

    def read_csv(path: str, expected_columns: List[str]) -> List[Dict[str, Any]]:
        if not os.path.isfile(path):
            errors.append(f"Файл не найден: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or list(reader.fieldnames) != expected_columns:
                errors.append(f"Неверные колонки в {path}. Ожидалось: {expected_columns}, найдено: {reader.fieldnames}")
                return []
            return list(reader)

    train_rows = read_csv(train_path, ["image_id", "x", "y", "w", "h", "vehicle_id"])
    query_rows = read_csv(query_path, ["image_id", "x", "y", "w", "h"])
    gallery_rows = read_csv(gallery_path, ["image_id", "x", "y", "w", "h"])

    all_image_ids: List[str] = []
    areas: List[int] = []

    def check_rows(rows: List[Dict[str, Any]], label: str) -> None:
        for row in rows:
            try:
                image_id = row["image_id"]
                all_image_ids.append(image_id)

                img_path = os.path.join(images_dir, image_id)
                if not os.path.isfile(img_path):
                    errors.append(f"{label}: файл изображения не найден для {image_id}")

                x = int(row["x"])
                y = int(row["y"])
                w = int(row["w"])
                h = int(row["h"])

                if x < 0 or y < 0:
                    errors.append(f"{label}: отрицательные координаты у {image_id}")
                if w <= 0 or h <= 0:
                    errors.append(f"{label}: ширина или высота бокса <= 0 у {image_id}")
                if x + w > cfg.image_width or y + h > cfg.image_height:
                    errors.append(f"{label}: бокс выходит за границы изображения у {image_id}")

                areas.append(w * h)
            except Exception as exc:
                errors.append(f"{label}: ошибка парсинга строки {row}: {exc}")

    check_rows(train_rows, "train.csv")
    check_rows(query_rows, "test_query.csv")
    check_rows(gallery_rows, "test_gallery.csv")

    # Дубликаты.
    id_counter = Counter(all_image_ids)
    duplicates = [k for k, v in id_counter.items() if v > 1]
    if duplicates:
        errors.append(f"Найдены дубликаты image_id: {duplicates[:10]}")

    # Минимальное число изображений на обучающую машину.
    train_counts = Counter()
    for row in train_rows:
        try:
            train_counts[int(row["vehicle_id"])] += 1
        except Exception:
            errors.append(f"train.csv: некорректный vehicle_id в строке {row}")

    for vid, count in train_counts.items():
        if count < 2:
            errors.append(f"train vehicle_id {vid} имеет меньше 2 изображений")

    train_ids = set(train_counts.keys())

    # Ground truth.
    ground_truth: Dict[str, int] = {}
    if os.path.isfile(gt_path):
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                raw_gt = json.load(f)
            ground_truth = {str(k): int(v) for k, v in raw_gt.items()}
        except Exception as exc:
            errors.append(f"Не удалось прочитать test_ground_truth.json: {exc}")
    else:
        errors.append("Файл meta/test_ground_truth.json не найден")

    test_ids = set(ground_truth.values())

    # Пересечение train/test.
    overlap = train_ids.intersection(test_ids)
    if overlap:
        errors.append(f"Тестовые идентичности пересекаются с обучающими: {sorted(list(overlap))[:10]}")

    # Проверка наличия всех тестовых изображений в ground truth.
    query_counts = Counter()
    gallery_counts = Counter()

    query_ids = set()
    gallery_ids = set()

    for row in query_rows:
        image_id = row["image_id"]
        query_ids.add(image_id)
        vid = ground_truth.get(image_id)
        if vid is None:
            errors.append(f"test_query image_id {image_id} отсутствует в ground truth")
        else:
            query_counts[vid] += 1

    for row in gallery_rows:
        image_id = row["image_id"]
        gallery_ids.add(image_id)
        vid = ground_truth.get(image_id)
        if vid is None:
            errors.append(f"test_gallery image_id {image_id} отсутствует в ground truth")
        else:
            gallery_counts[vid] += 1

    for image_id in ground_truth.keys():
        if image_id not in query_ids and image_id not in gallery_ids:
            errors.append(f"Ground truth содержит {image_id}, но изображения нет в test_query/test_gallery CSV")

    # Для каждой тестовой машины нужен хотя бы один query и один gallery.
    for vid in test_ids:
        if query_counts.get(vid, 0) < 1:
            errors.append(f"Тестовая идентичность {vid} не имеет ни одного query изображения")
        if gallery_counts.get(vid, 0) < 1:
            errors.append(f"Тестовая идентичность {vid} не имеет ни одного gallery изображения")

    # Общая проверка, что датасет не пуст.
    if len(train_rows) == 0:
        errors.append("train.csv пуст")
    if len(query_rows) == 0:
        errors.append("test_query.csv пуст")
    if len(gallery_rows) == 0:
        errors.append("test_gallery.csv пуст")

    total_images = len(train_rows) + len(query_rows) + len(gallery_rows)
    train_vehicles = len(train_counts)
    test_vehicles = len(test_ids)

    avg_images_per_vehicle = 0.0
    if train_vehicles + test_vehicles > 0:
        avg_images_per_vehicle = total_images / float(train_vehicles + test_vehicles)

    avg_bbox_area = 0.0
    if areas:
        avg_bbox_area = float(np.mean(areas))

    stats = {
        "total_images": total_images,
        "train_images": len(train_rows),
        "query_images": len(query_rows),
        "gallery_images": len(gallery_rows),
        "train_vehicles": train_vehicles,
        "test_vehicles": test_vehicles,
        "avg_images_per_vehicle": avg_images_per_vehicle,
        "avg_bbox_area": avg_bbox_area,
    }

    return len(errors) == 0, errors, stats


# ============================================================
# Очистка мира
# ============================================================

def cleanup_carla(
    world: Optional[carla.World],
    original_settings: Optional[Any],
    original_weather: Optional[Any],
    tm: Optional[Any],
) -> None:
    """Удалить созданные актёры и восстановить исходные настройки мира."""
    if world is None:
        return

    # Удаляем актёров, созданных скриптом, по роли.
    try:
        actors = list(world.get_actors())
        for actor in actors:
            try:
                role = actor.attributes.get("role_name", "") if hasattr(actor, "attributes") else ""
                if role.startswith("reid_"):
                    try:
                        if hasattr(actor, "stop"):
                            actor.stop()
                    except Exception:
                        pass
                    safe_destroy(actor)
            except Exception:
                pass
    except Exception as exc:
        log(f"Ошибка при очистке актёров: {exc}")

    if tm is not None:
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass

    if original_settings is not None:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    if original_weather is not None:
        try:
            world.set_weather(original_weather)
        except Exception:
            pass


# ============================================================
# main
# ============================================================

def main() -> int:
    """Главная функция."""

    @dataclass
    class Config:
        # Подключение.
        host: str = "localhost"
        port: int = 2000
        traffic_manager_port: int = 8000
        connection_timeout: float = 20.0
        map_load_wait: float = 8.0

        # Мир.
        town: str = "Town10HD_Opt"
        synchronous_mode: bool = True
        fixed_delta_seconds: float = 0.05
        seed: int = 42

        # Выход.
        output_dir: str = "generated_dataset"

        # Камеры.
        image_width: int = 1920
        image_height: int = 1080
        camera_fov: float = 90.0

        # Датасет.
        num_train_identities: int = 200
        min_train_images_per_identity: int = 15
        max_train_images_per_identity: int = 30

        num_test_identities: int = 30
        num_test_query_per_identity: int = 3
        num_test_gallery_per_identity: int = 10

        train_id_start: int = 1
        test_id_start: int = 100001

        # Фильтры боксов.
        min_bbox_area: int = 120 * 120
        min_visible_fraction: float = 0.65
        max_bbox_area_fraction: float = 0.95

        # Съёмка.
        max_images_per_camera_per_identity: int = 12
        capture_cooldown_frames: int = 12
        jpeg_quality: int = 95

        # Проверка видимости через доступный программный fallback.
        use_visibility_raycast_fallback: bool = True

        # Доступные модели машин. Если недоступны, будут использованы другие четырёхколёсные.
        blueprint_filter: List[str] = field(
            default_factory=lambda: [
                "vehicle.audi.a2",
                "vehicle.audi.tt",
                "vehicle.bmw.grandtourer",
                "vehicle.citroen.c3",
                "vehicle.ford.mustang",
                "vehicle.lincoln.mkz_2017",
                "vehicle.mercedes.coupe",
                "vehicle.nissan.micra",
                "vehicle.seat.leon",
                "vehicle.tesla.model3",
                "vehicle.toyota.prius",
                "vehicle.volkswagen.t2",
            ]
        )

        # Надёжность.
        max_spawn_attempts: int = 3
        spawn_attempts_per_vehicle: int = 15
        max_ticks_per_phase: int = 1200
        max_failed_ticks: int = 150
        sensor_queue_timeout: float = 10.0

        # Погода и блоки съёмки.
        weather_variants_per_identity: int = 4
        images_per_weather_chunk: int = 6
        max_capture_retries: int = 3

    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Создаём выходные директории.
    try:
        os.makedirs(os.path.join(cfg.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "meta"), exist_ok=True)
    except Exception as exc:
        print(f"Не удалось создать выходные директории: {exc}")
        return 1

    client: Optional[carla.Client] = None
    world: Optional[carla.World] = None
    tm: Optional[Any] = None
    original_settings: Optional[Any] = None
    original_weather: Optional[Any] = None

    exit_code = 1
    start_time = time.time()

    try:
        client = connect_client(cfg.host, cfg.port, cfg.connection_timeout)
        world, original_settings, original_weather = setup_world(client, cfg)
        tm = setup_traffic_manager(client, cfg)

        blueprints = get_available_vehicle_blueprints(world, cfg)
        log(f"Доступно моделей автомобилей: {len(blueprints)}")

        train_identities, test_identities = generate_identity_specs(cfg, blueprints)
        log(
            f"Идентичности: train={len(train_identities)}, test={len(test_identities)}"
        )

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("На карте нет spawn points для автомобилей.")

        counters = {"discarded": 0}
        master_rng = random.Random(cfg.seed + 999)

        train_records: List[Dict[str, Any]] = []
        query_records: List[Dict[str, Any]] = []
        gallery_records: List[Dict[str, Any]] = []
        ground_truth: Dict[str, int] = {}
        successful_identities: List[Identity] = []

        # Обучение.
        for idx, identity in enumerate(train_identities, start=1):
            log(f"Train identity {idx}/{len(train_identities)}, id={identity.identity_id}")
            records = process_train_identity(world, identity, spawn_points, cfg, counters, master_rng)

            if records:
                train_records.extend(records)
                successful_identities.append(identity)
                log(f"  сохранено train изображений: {len(records)}")
            else:
                log("  идентичность пропущена из-за недостатка удачных кадров")

        # Тест.
        for idx, identity in enumerate(test_identities, start=1):
            log(f"Test identity {idx}/{len(test_identities)}, id={identity.identity_id}")
            q_records, g_records = process_test_identity(
                world,
                identity,
                spawn_points,
                cfg,
                counters,
                master_rng,
            )

            if q_records and g_records:
                query_records.extend(q_records)
                gallery_records.extend(g_records)
                successful_identities.append(identity)

                for rec in q_records + g_records:
                    ground_truth[rec["image_id"]] = identity.identity_id

                log(f"  сохранено query={len(q_records)}, gallery={len(g_records)}")
            else:
                log("  тестовая идентичность пропущена из-за недостатка удачных кадров")

        elapsed = time.time() - start_time
        extra_info = {
            "elapsed_seconds": elapsed,
            "discarded_frames": counters["discarded"],
        }

        write_datasets(
            cfg=cfg,
            train_records=train_records,
            query_records=query_records,
            gallery_records=gallery_records,
            ground_truth=ground_truth,
            successful_identities=successful_identities,
            extra_info=extra_info,
        )

        ok, errors, stats = validate_dataset(cfg)

        if not ok:
            log("Валидация датасета не пройдена:")
            for err in errors[:30]:
                log(f" - {err}")
            if len(errors) > 30:
                log(f" ... и ещё {len(errors) - 30} ошибок")
            exit_code = 1
        else:
            log("Валидация датасета пройдена успешно.")
            log("Статистика:")
            log(f"  всего изображений:              {stats['total_images']}")
            log(f"  train изображений:              {stats['train_images']}")
            log(f"  query изображений:              {stats['query_images']}")
            log(f"  gallery изображений:            {stats['gallery_images']}")
            log(f"  обучающих машин:                {stats['train_vehicles']}")
            log(f"  тестовых машин:                 {stats['test_vehicles']}")
            log(f"  среднее изображений на машину:  {stats['avg_images_per_vehicle']:.2f}")
            log(f"  средняя площадь бокса:          {stats['avg_bbox_area']:.1f}")
            log(f"  отброшенных кадров:             {counters['discarded']}")
            exit_code = 0

    except KeyboardInterrupt:
        log("Остановлено пользователем (Ctrl+C).")
        exit_code = 130
    except Exception as exc:
        log(f"Критическая ошибка: {exc}")
        traceback.print_exc()
        exit_code = 1
    finally:
        cleanup_carla(world, original_settings, original_weather, tm)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())