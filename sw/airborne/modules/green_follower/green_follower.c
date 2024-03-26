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

#define GREEN_FOLLOWER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[green_follower->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if GREEN_FOLLOWER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// define settings
float gf_set_speed = 0.4f;           // max flight speed [m/s]
float gf_floor_count_frac = 0.01f;  // percentage of the image that needs to be green before turning around
float gf_sideways_speed_factor = 0.25f;  // Oversteer correction
float gf_set_height = 0.5f;
float gf_height_gain = 0.1f;

// define and initialise global variables
float current_best_heading_green = 0.0f;              // heading with the longest available floor space in [rad], where 0 is ahead, positive is right
float current_best_heading_corr = 0.0f;              // heading with the longest available floor space in [rad], where 0 is ahead, positive is right
float current_safe_length = 0.0f;
uint32_t current_green_pixels = 0;
uint8_t waiting_cycles = 0;

float previous_best_heading = 0.0f;
int escaping = 0;


// This call back will be used to receive the color count from the orange detector
#ifndef GREEN_FOLLOWER_VISUAL_DETECTION_ID
#define GREEN_FOLLOWER_VISUAL_DETECTION_ID ABI_BROADCAST
#error Please define GREEN_FOLLOWER_VISUAL_DETECTION_ID to be GREEN_DETECTION_ID
#endif
static abi_event green_detection_ev;
static void green_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               float __attribute__((unused)) best_heading, float __attribute__((unused)) safe_length,
                               uint32_t __attribute__((unused)) green_pixels)
{
    current_best_heading_green = best_heading;
    current_safe_length = safe_length;
    current_green_pixels = green_pixels;
}

#ifndef CORR_DEPTH_FINDER_VISUAL_DETECTION_ID
#define CORR_DEPTH_FINDER_VISUAL_DETECTION_ID ABI_BROADCAST
#error Please define CORR_DEPTH_FINDER_VISUAL_DETECTION_ID to be CORR_DEPTH_ID
#endif
static abi_event corr_depth_ev;
static void corr_depth_cb(uint8_t __attribute__((unused)) sender_id,
                          float __attribute__((unused)) best_heading)
{
  current_best_heading_corr = best_heading;
}

/*
 * Initialisation function
 */
void green_follower_init(void)
{
    // bind our green_detector callbacks to receive the heading outputs
    AbiBindMsgGREEN_DETECTION(GREEN_FOLLOWER_VISUAL_DETECTION_ID, &green_detection_ev, green_detection_cb);
    AbiBindMsgDEPTH_FINDER_HEADING(CORR_DEPTH_FINDER_VISUAL_DETECTION_ID, &corr_depth_ev, corr_depth_cb);
}

/*
 * Function that moves towards the area with the most green
 */
void green_follower_periodic(void)
{

    uint32_t floor_count_threshold = (uint32_t) (gf_floor_count_frac * 520.0 * 240.0);

    // Only run the module if we are in the correct flight mode
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
        current_best_heading_green = 0.0f;
        return;
    }

//    VERBOSE_PRINT("GF: Current best heading: %f\nCurrent safe length: %f\n", current_best_heading_green, current_safe_length);
//    VERBOSE_PRINT("GF: floor threshold: %d / %d\n", current_green_pixels, floor_count_threshold);
    //float percentage = current_green_pixels*100/(520*240);
    //VERBOSE_PRINT("GF: floor threshold: %d / %d = %f\n", current_green_pixels, floor_count_threshold,percentage);
    /*if (waiting_cycles > 0){
        guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + M_PI/20);
        waiting_cycles--;
    }
    else if (current_green_pixels > floor_count_threshold) {*/
    if (current_green_pixels > floor_count_threshold && current_safe_length > 15) {
      //if (escaping == 1){
      //  escaping = 0;
      //  previous_best_heading = 0;
      //}
      float speed_sp = fminf(gf_set_speed, current_safe_length / 100);

        //VERBOSE_PRINT("GF: Moving from %f towards %f (d=%f) at %f\n", stateGetNedToBodyEulers_f()->psi, stateGetNedToBodyEulers_f()->psi + current_best_heading_green, current_best_heading_green, speed_sp);
      float heading_sp = current_best_heading_green + current_best_heading_corr;
      //heading_sp = 0.7*previous_best_heading + 0.3*heading_sp;
      if (fabs(heading_sp) <= M_PI/6) {
        if (fabs(heading_sp-previous_best_heading)<=15*M_PI/180) {
          guidance_h_set_body_vel(speed_sp, 0);
          guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + 0);
          previous_best_heading = 0;
        } else {
          heading_sp = 0.5*previous_best_heading + 0.5*heading_sp;
          guidance_h_set_body_vel(speed_sp, gf_sideways_speed_factor * heading_sp);
          guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + heading_sp);
          previous_best_heading = heading_sp;
        }
      } else {
        if (fabs(heading_sp-previous_best_heading) < M_PI/2) {
          guidance_h_set_body_vel(0.5f * speed_sp, gf_sideways_speed_factor * heading_sp);
          guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + heading_sp);
          previous_best_heading = heading_sp;
        } else {
          guidance_h_set_body_vel(0.5f * speed_sp, 0);
          guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + previous_best_heading);
          previous_best_heading = previous_best_heading;
        }
      }


      //guidance_h_set_body_vel(speed_sp, gf_sideways_speed_factor * heading_sp);

      // Height controller doesn't work yet
      // guidance_v_set_vz((gf_set_height - stateGetPositionNed_f()->z) * gf_height_gain);
      // VERBOSE_PRINT("h: %f, u: %f", stateGetPositionNed_f()->z, (gf_set_height - stateGetPositionNed_f()->z) * gf_height_gain);
    } else {
      // VERBOSE_PRINT("GF: ESCAPING! Floor threshold: %d / %d\n", current_green_pixels, floor_count_threshold);
      float heading_sp = M_PI/4;
      guidance_h_set_body_vel(0.0f, 0);
      guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi + heading_sp);
      previous_best_heading = 0;
      //escaping = 1;
      // guidance_v_set_vz(0);
      // waiting_cycles = 4;
    }

    return;
}