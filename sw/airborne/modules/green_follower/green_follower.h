//
// Created by lilly on 6-3-24.
//

#ifndef PAPARAZZI_GREEN_FOLLOWER_H
#define PAPARAZZI_GREEN_FOLLOWER_H

#endif //PAPARAZZI_GREEN_FOLLOWER_H

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
extern float gf_green_threshold;  // obstacle detection threshold as a fraction of total of image
extern float gf_floor_count_frac; // floor detection threshold as a fraction of total of image
extern float gf_set_speed;        // max flight speed [m/s]

extern void green_follower_init(void);
extern void green_follower_periodic(void);

#endif

