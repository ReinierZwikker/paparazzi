//
// Created by lilly on 6-3-24.
//

#ifndef PAPARAZZI_GREEN_DETECTOR_H
#define PAPARAZZI_GREEN_DETECTOR_H

#include <stdint.h>

// Module settings
extern uint8_t gd_lum_min;
extern uint8_t gd_lum_max;
extern uint8_t gd_cb_min;
extern uint8_t gd_cb_max;
extern uint8_t gd_cr_min;
extern uint8_t gd_cr_max;

extern void green_detector_init(void);
extern void green_detector_periodic(void);

#endif //PAPARAZZI_GREEN_DETECTOR_H
