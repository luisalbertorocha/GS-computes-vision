import cv2
import numpy as np

from src.preprocess import steps
from src.privacy.blur_faces_plates import anonymize, blur_regions


def test_letterbox_keeps_aspect_and_pads(road_image):
    out, scale, (pad_x, pad_y) = steps.letterbox(road_image, 640)
    assert out.shape == (640, 640, 3)
    assert scale == 0.5
    assert (pad_x, pad_y) == (0, 140)
    assert (out[0, 0] == steps.PAD_COLOR).all()  # borda cinza
    assert abs(int(out[320, 320, 0]) - 120) < 30  # conteúdo no meio


def test_letterbox_upscales_small_images():
    out, scale, pad = steps.letterbox(np.zeros((100, 200, 3), np.uint8), 640)
    assert out.shape == (640, 640, 3) and scale == 3.2 and pad == (0, 160)


def test_clahe_increases_contrast_without_changing_shape():
    rng = np.random.default_rng(0)
    low_contrast = rng.integers(100, 120, (200, 200, 3), dtype=np.uint8)
    out = steps.clahe_lab(low_contrast)
    assert out.shape == low_contrast.shape
    assert out.std() > low_contrast.std()


def test_bilateral_reduces_noise(road_image):
    patch = road_image[:200, :200]
    assert steps.denoise_bilateral(patch).std() < patch.std()


def test_roi_mask_covers_bottom_and_skips_padding(road_image):
    boxed, scale, (pad_x, pad_y) = steps.letterbox(road_image, 640)
    content = (pad_x, pad_y, pad_x + 640, pad_y + 360)
    mask = steps.road_roi_mask(boxed.shape, content)
    assert mask[pad_y + 10, 320] == 0  # topo da cena fora da ROI
    assert mask[pad_y + 300, 320] == 255  # pista dentro
    assert mask[:pad_y].max() == 0 and mask[pad_y + 360 :].max() == 0  # padding fora


def test_canny_morphology_finds_the_hole(road_image):
    boxed, _, _ = steps.letterbox(road_image, 640)
    mask = steps.canny_morphology(steps.denoise_bilateral(steps.clahe_lab(boxed)))
    # centro do buraco em coordenadas letterbox: (640*0.5, 520*0.5 + 140)
    assert mask[400, 320] == 255
    assert mask[200, 100] == 0


def test_warp_birdseye_keeps_shape(road_image):
    assert steps.warp_birdseye(road_image).shape == road_image.shape


def test_blur_regions_only_touches_the_region():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    out = blur_regions(img, [(100, 100, 60, 60, "rosto")], expand=0.0)
    assert not np.array_equal(out[100:160, 100:160], img[100:160, 100:160])
    assert np.array_equal(out[:100], img[:100])
    assert np.array_equal(out[200:], img[200:])


def test_anonymize_leaves_plain_image_untouched_and_cascades_load():
    flat = np.full((240, 320, 3), 128, np.uint8)
    out, regions = anonymize(flat)
    assert regions == [] and np.array_equal(out, flat)
    assert cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    ).empty() is False
