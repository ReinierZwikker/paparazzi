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
#include <math.h>
#include <time.h>

#define GREEN_FOLLOWER TRUE

#define PRINT(string,...) fprintf(stderr, "[green_follower->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if GREEN_FOLLOWER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// define settings
//float gf_floor_count_frac = 0.05f; // floor detection threshold as a fraction of total of image
float gf_set_speed = 1.0f;           // max flight speed [m/s]
float oag_floor_count_frac = 0.05f;  // percentage of the image that needs to be green before turning around

// define and initialise global variables
float current_best_heading = 0.0f;              // heading with the longest available floor space in [rad], where 0 is ahead, positive is right
float current_safe_length = 0.0f;
uint8_t current_green_pixels = 0;


// This call back will be used to receive the color count from the orange detector
#ifndef GREEN_FOLLOWER_VISUAL_DETECTION_ID
#error Please define GREEN_FOLLOWER_VISUAL_DETECTION_ID to be GREEN_DETECTION_ID
#endif
static abi_event green_detection_ev;
static void green_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               float __attribute__((unused)) best_heading, float __attribute__((unused)) safe_length,
                               uint8_t __attribute__((unused)) green_pixels)
{
    current_best_heading = best_heading;
    current_safe_length = safe_length;
    current_green_pixels = green_pixels;
}

/*
 * Initialisation function
 */
void green_follower_init(void)
{
    // Initialise random values
    srand(time(NULL));
    chooseRandomIncrementAvoidance();

    // bind our green_detector callbacks to receive the heading outputs
    AbiBindMsgGREEN_DETECTION(GREEN_FOLLOWER_VISUAL_DETECTION_ID, &green_detection_ev, green_detection_cb);
}

/*
 * Function that moves towards the area with the most green
 */
void green_follower_periodic(void)
{

    int32_t floor_count_threshold = oag_floor_count_frac * 520.0 * 240.0;

    // Only run the module if we are in the correct flight mode
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
        current_best_heading = 0.0f
        return;
    }

    if (current_green_pixels > floor_count_threshold) {
        float speed_sp = fminf(gf_set_speed, current_safe_length / 100);

        guidance_h_set_body_vel(speed_sp, 0);
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + current_best_heading);
    }
    else {
        guidance_h_set_body_vel(0, 0);
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + M_PI/2);
    }

    return;
}