//
// Created by lilly on 6-3-24.
//


/*
 * ADAPTED FROM ORANGE FOLLOWER
 * Orange follower is Copyright (C) Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/green_follower/green_follower.c"
 * @author
 * TODO
 * Moving towards most green, floor counter is not included yet
 */

#include "modules/green_follower/green_follower.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <stdio.h>
#include <time.h>

#define GREEN_FOLLOWER TRUE

#define PRINT(string,...) fprintf(stderr, "[green_follower->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if GREEN_FOLLOWER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// define settings
float gf_green_threshold = 0.0f;  // obstacle detection threshold as a fraction of total of image
//float gf_floor_count_frac = 0.05f; // floor detection threshold as a fraction of total of image
float gf_set_speed = 1.0f;        // max flight speed [m/s]

// define and initialise global variables
float best_heading = 0.0f;              // heading with the longest available floor space in [rad], where 0 is ahead, positive is right
//int32_t floor_count = 0;                // green color count from color filter for floor detection
//int32_t floor_centroid = 0;             // floor detector centroid in y direction (along the horizon)


// This call back will be used to receive the color count from the orange detector
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#error This module requires two color filters, as such you have to define ORANGE_AVOIDER_VISUAL_DETECTION_ID to the orange filter
#error Please define ORANGE_AVOIDER_VISUAL_DETECTION_ID to be COLOR_OBJECT_DETECTION1_ID or COLOR_OBJECT_DETECTION2_ID in your airframe
#endif
static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t __attribute__((unused)) pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
//    color_count = quality;
}

//#ifndef FLOOR_VISUAL_DETECTION_ID
//#error This module requires two color filters, as such you have to define FLOOR_VISUAL_DETECTION_ID to the orange filter
//#error Please define FLOOR_VISUAL_DETECTION_ID to be COLOR_OBJECT_DETECTION1_ID or COLOR_OBJECT_DETECTION2_ID in your airframe
//#endif
//static abi_event floor_detection_ev;
//static void floor_detection_cb(uint8_t __attribute__((unused)) sender_id,
//                               int16_t __attribute__((unused)) pixel_x, int16_t pixel_y,
//                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
//                               int32_t quality, int16_t __attribute__((unused)) extra)
//{
//    floor_count = quality;
//    floor_centroid = pixel_y;
//}

/*
 * Initialisation function
 */
void green_follower_init(void)
{
    // Initialise random values
    srand(time(NULL));
    chooseRandomIncrementAvoidance();

    // bind our colorfilter callbacks to receive the color filter outputs
    AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
//    AbiBindMsgVISUAL_DETECTION(FLOOR_VISUAL_DETECTION_ID, &floor_detection_ev, floor_detection_cb);
}

/*
 * Function that moves towards the area with the most green
 */
void green_follower_periodic(void)
{
    // Only run the mudule if we are in the correct flight mode
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
        best_heading = 0.0f
        return;
    }

    // compute current color thresholds
//    int32_t floor_count_threshold = oag_floor_count_frac * front_camera.output_size.w * front_camera.output_size.h;
//    float floor_centroid_frac = floor_centroid / (float)front_camera.output_size.h / 2.f;

    VERBOSE_PRINT("Color_count: %d  threshold: %d state: %d \n", color_count, color_count_threshold, navigation_state);
    VERBOSE_PRINT("Floor count: %d, threshold: %d\n", floor_count, floor_count_threshold);
    VERBOSE_PRINT("Floor centroid: %f\n", floor_centroid_frac);


    float speed_sp = fminf(gf_max_speed, 0); // TODO Set to floor length of longest heading

    guidance_h_set_body_vel(speed_sp, 0);
    guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);

    return;
}