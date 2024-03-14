/*
 * Copyright (C) 2015
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */

/**
 * @file modules/computer_vision/video_capture.h
 */

#ifndef VIDEO_CAPTURE_H_
#define VIDEO_CAPTURE_H_

#include <stdbool.h>

// Module settings
extern bool video_capture_take_shot;
extern bool video_capture_record_video;

// Module structures
/* The different type of images we currently support */
enum image_type {
    IMAGE_YUV422,     ///< UYVY format (uint16 per pixel)
    IMAGE_GRAYSCALE,  ///< Grayscale image with only the Y part (uint8 per pixel)
    IMAGE_JPEG,       ///< An JPEG encoded image (not per pixel encoded)
    IMAGE_GRADIENT,    ///< An image gradient (int16 per pixel)
    IMAGE_INT16     ///< An image to hold disparity image data from openCV (int16 per pixel)
};

/* Main image structure */
struct image_t {
    enum image_type type;   ///< The image type
    uint16_t w;             ///< Image width
    uint16_t h;             ///< Image height
    struct timeval ts;      ///< The timestamp of creation
    struct FloatEulers eulers;   ///< Euler Angles at time of image
    uint32_t pprz_ts;       ///< The timestamp in us since system startup

    uint8_t buf_idx;        ///< Buffer index for V4L2 freeing
    uint32_t buf_size;      ///< The buffer size
    void *buf;              ///< Image buffer (depending on the image_type)
};

// Module functions
extern void video_capture_init(void);
extern void video_capture_shoot(void); // Capture single image
extern void video_capture_start_capture(void); // Start video capture
extern void video_capture_stop_capture(void); // Stop video capture
extern void video_capture_save(struct image_t *img); // Save current image

#endif /* VIDEO_CAPTURE_H_ */
