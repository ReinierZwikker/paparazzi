//
// Created by lilly on 17-2-24.
//

#include "modules/my_first_module/circler.h"
#include "state.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "stabilization/stabilization_attitude_rc_setpoint.h"
#include "stabilization/stabilization_attitude.h"


#define _USE_MATH_DEFINES
#include "math.h"

//#include "GL/gl.h"


#define CIRCLER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[circler->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if CIRCLER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

float set_speed = 3.0f; // [m/s]
float set_radius = 2.0f; // [m]

enum hop_state_t {
    HOP,
    FLIP,
    FLIP_BACK,
    CATCH
};

hop_state_t hop_state;

float k_radius = 0.6f; // [m]

struct CylCoor_f {
    float theta;
    float radius;
    float altitude;
};

void circler_init(void) {

    // create a new texture name
    // bind the texture name to a texture target
//    glBindTexture(GL_TEXTURE_2D,1);
//    // turn off filtering and set proper wrap mode
//    // (obligatory for float textures atm)
//    glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//    glTexParameteri(texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP);
//    glTexParameteri(texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP);
//    // set texenv to replace instead of the default modulate
//    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
//    // and allocate graphics memory
//    glTexImage2D(texture_target, 0, internal_format,
//                 texSize, texSize, 0, texture_format, GL_FLOAT, 0);

    return;
}

void convert_to_cylindrical_coords(struct NedCoor_f *ned_position, struct CylCoor_f *cyl_position) {
    cyl_position->theta = atan2f(ned_position->y, ned_position->x);
    cyl_position->radius = sqrtf(powf(ned_position->x, 2) + powf(ned_position->y, 2));
    cyl_position->altitude = -ned_position->z;
}

void fly_circle(void) {
    VERBOSE_PRINT("IN CIRCLER MODE \n");
    struct NedCoor_f *current_ned_pos = stateGetPositionNed_f();
//    VERBOSE_PRINT("Current position: %f, %f, %f \n", current_ned_pos->x, current_ned_pos->y, current_ned_pos->z);

    struct CylCoor_f current_cyl_pos;
    convert_to_cylindrical_coords(current_ned_pos, &current_cyl_pos);

    VERBOSE_PRINT("Current position: t=%f, r=%f, a=%f \n", current_cyl_pos.theta / M_PI * 180, current_cyl_pos.radius,
                  current_cyl_pos.altitude);

    guidance_h_set_body_vel(set_speed, 0);

    float target_heading = current_cyl_pos.theta + 0.5 * M_PI;

    target_heading -= k_radius * (set_radius - current_cyl_pos.radius);

    VERBOSE_PRINT("Target heading: %f \n", target_heading / M_PI * 180);


    guidance_h_set_heading(target_heading);

    return;
}

void fly_radar_hop(void) {
    VERBOSE_PRINT("IN CIRCLER MODE \n");
    struct NedCoor_f *current_ned_pos = stateGetPositionNed_f();
    struct Int32Eulers *current_euler_angles = stateGetNedToBodyEulers_i();
    switch (hop_state) {
        case HOP:
            autopilot_static_set_mode(2);

            stabilization_cmd[COMMAND_THRUST] = 8000; // Thrust to go up first
            if (current_ned_pos->z > 4) {
                hop_state = FLIP;
            }
            break;
        case FLIP:
            stabilization_cmd[COMMAND_THRUST] = 0; // Enter free fall
            stabilization_cmd[COMMAND_PITCH]  = 4000;
            if (current_euler_angles->theta > 0.25 * INT32_ANGLE_PI)
                hop_state = FLIP_BACK;
            break;
        case FLIP_BACK:
            stabilization_cmd[COMMAND_THRUST] = 0; // Enter free fall
            stabilization_cmd[COMMAND_PITCH]  = -4000;
            if (current_euler_angles->theta < 0.25 * INT32_ANGLE_PI)
                hop_state = CATCH;
            break;
        case CATCH:
            autopilot_static_set_mode(4);
            break;
    }

    }

void circler_periodic(void) {

    // Only run the AP if we are in the correct flight mode
    if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
        return;
    }

    fly_circle();

//    fly_radar_hop();


}

