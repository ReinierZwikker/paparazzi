//
// Created by lilly on 6-3-24.
//

#ifndef PAPARAZZI_GREEN_FOLLOWER_H
#define PAPARAZZI_GREEN_FOLLOWER_H


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
 */

// settings
extern float gf_set_speed;        // max flight speed [m/s]
extern float gf_floor_count_frac; // Ground fraction before turning

extern void green_follower_init(void);
extern void green_follower_periodic(void);

#endif //PAPARAZZI_GREEN_FOLLOWER_H

