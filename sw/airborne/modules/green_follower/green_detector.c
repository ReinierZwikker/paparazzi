//
// Copied from lilly's v1 and modified
//

#include "modules/green_follower/green_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

// #ifndef CACHE_LINE_LENGTH
// #define CACHE_LINE_LENGTH 64
// #endif

#ifndef IMAGE_BOOL
#define IMAGE_BOOL 5
#endif

#define PAINT_OVER_IMAGE TRUE

#ifndef GREENFILTER_FPS
#define GREENFILTER_FPS 0       ///< Default FPS (zero means run at camera fps)
#endif
PRINT_CONFIG_VAR(COLORFILTER_FPS)

#define GREEN_DETECTOR_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[green_follower->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if GREEN_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// Filter Settings
uint8_t gd_lum_min = 60;
uint8_t gd_lum_max = 110;
uint8_t gd_cb_min = 75;
uint8_t gd_cb_max = 110;
uint8_t gd_cr_min = 110;
uint8_t gd_cr_max = 130;
const uint8_t kernel_size_w = 10;	// Note: Needs to be an integer divider of the input image pixel width 
const uint8_t kernel_size_h = 20; // Note: Needs to be an integer divider of the input image pixel height
// NOTE [Aaron]: If we make it such that the number of pixels in the reduced image (after the filter) is divisible by 8, 
// we could do bitwise boolean allocation to save on memory, but waaay overkill for now
const float ray_weights[] = {0.1f, 0.5f, 0.85f, 1.0f, 0.85f, 0.5f, 0.1f};

const float ray_angles[] = {M_PI/6.0f, 
														M_PI/6.0f + 			 M_PI/9.0f, 
														M_PI/6.0f + 2.0f * M_PI/9.0f, 
														M_PI/6.0f + 3.0f * M_PI/9.0f, 
														M_PI/6.0f + 4.0f * M_PI/9.0f, 
														M_PI/6.0f + 5.0f * M_PI/9.0f, 
														M_PI/6.0f + 6.0f * M_PI/9.0f};


static pthread_mutex_t mutex;

struct heading_object_t {
    float best_heading;
    float safe_length;
    uint32_t green_pixels;
    bool updated;
};
struct heading_object_t global_heading_object;


void get_direction(struct image_t* original_image, float* best_heading, float* safe_length, uint32_t* green_pixels);


/*
 * object_detector
 * @param img - input image to process
 * @return img
 */
static struct image_t *green_heading_finder(struct image_t *img) {

    float best_heading, safe_length;
    uint32_t green_pixels = 0;

		get_direction(img, &best_heading, &safe_length, &green_pixels);

    pthread_mutex_lock(&mutex);
    global_heading_object.best_heading = best_heading;
    global_heading_object.safe_length = safe_length;
    global_heading_object.green_pixels = green_pixels;
    global_heading_object.updated = true;
    pthread_mutex_unlock(&mutex);
    return img;
}

struct image_t *green_heading_finder1(struct image_t *img, uint8_t camera_id);
struct image_t *green_heading_finder1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
    return green_heading_finder(img);
}

void green_detector_init(void) {
    memset(&global_heading_object, 0, sizeof(struct heading_object_t));
    pthread_mutex_init(&mutex, NULL);

    #ifdef GREEN_DETECTOR_LUM_MIN
        gd_lum_min = GREEN_DETECTOR_LUM_MIN;
        gd_lum_max = GREEN_DETECTOR_LUM_MAX;
        gd_cb_min = GREEN_DETECTOR_CB_MIN;
        gd_cb_max = GREEN_DETECTOR_CB_MAX;
        gd_cr_min = GREEN_DETECTOR_CR_MIN;
        gd_cr_max = GREEN_DETECTOR_CR_MAX;
    #endif

    cv_add_to_device(&GREENFILTER_CAMERA, green_heading_finder1, GREENFILTER_FPS, 0);
}

void green_detector_periodic(void) {
    static struct heading_object_t local_heading_object;
    pthread_mutex_lock(&mutex);
    memcpy(&local_heading_object, &global_heading_object, sizeof(struct heading_object_t));
    pthread_mutex_unlock(&mutex);

    if(local_heading_object.updated){
        AbiSendMsgGREEN_DETECTION(GREEN_DETECTION_ID, local_heading_object.best_heading, local_heading_object.safe_length, local_heading_object.green_pixels);
        local_heading_object.updated = false;
    }
}

static void green_filter(struct image_t* original_image, struct image_t* filtered_image) {

	uint8_t *original_buffer = original_image->buf;
  uint8_t *filtered_buffer = filtered_image->buf;

	// Perform mean pooling on the original image
	for (uint16_t row=0; row<filtered_image->h; row++) {
		for (uint16_t col=0; col<filtered_image->w; col++) {

			uint32_t yuv_y = 0;
			uint32_t yuv_u = 0;
			uint32_t yuv_v = 0;

			// Sum color channels of pixels in the kernel
			for (uint8_t row_in_kernel=0; row_in_kernel<kernel_size_h; row_in_kernel++) {
				for (int8_t col_in_kernel=0; col_in_kernel<kernel_size_w; col_in_kernel++) {
					// Parse depending on even or uneven col nr
					if ((col+col_in_kernel) % 2 == 0) {
						// Even col nr
						yuv_u += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)];      // U
						yuv_y += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1];  // Y1
						yuv_v += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 2];  // V
					} 
					else {
						// Uneven col nr
						yuv_u += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) - 2];  // U
						yuv_v += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)];      // V
						yuv_y += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1];  // Y2
					}
				}
			}

			// Divide by number of pixels in the kernel to get the mean values
			yuv_y = yuv_y / (kernel_size_w * kernel_size_h);
			yuv_u = yuv_u / (kernel_size_w * kernel_size_h);
			yuv_v = yuv_v / (kernel_size_w * kernel_size_h);

			// Perform limit checks and assign value to filter's pixel
			if (	(yuv_y >= gd_lum_min) && (yuv_y < gd_lum_max) &&
						(yuv_u >= gd_cb_min) 	&& (yuv_u < gd_cb_max) && 
						(yuv_v >= gd_cr_min) 	&& (yuv_v < gd_cr_max)) {

				filtered_buffer[row * filtered_image->w + col] = true;

        #if PAINT_OVER_IMAGE
        // Go over all pixels
        for (uint8_t row_in_kernel=0; row_in_kernel<kernel_size_h; row_in_kernel++) {
          for (int8_t col_in_kernel=0; col_in_kernel<kernel_size_w; col_in_kernel++) {
            // Parse depending on even or uneven col nr
            if ((col+col_in_kernel) % 2 == 0) {
              // Even col nr
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1] = 200;  // Y1
            } 
            else {
              // Uneven col nr
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1] = 50;  // Y2
            }
          }
        }
        #endif

			}
			else {
				filtered_buffer[row * filtered_image->w + col] = false;
			}
		}
	}
}


void get_direction(struct image_t* original_image, float* best_heading, float* safe_length, uint32_t* green_pixels) {

	// Array where ray scores will be added
	float ray_scores[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	// Array which will contain the filtered image
  struct image_t filtered_image;
  image_create(&filtered_image, original_image->w / kernel_size_w, original_image->h / kernel_size_h, IMAGE_BOOL);
  // Ptr to the filtered buffer
  uint8_t *filtered_buffer = &(filtered_image.buf);

	// Apply filter to original image
	green_filter(original_image, &filtered_image);

	// Go through each pixel in the filtered image
	// and determine which ray it belongs to
	for (uint16_t row=0; row<filtered_image.h; row++) {
		for (uint16_t col=0; col<filtered_image.w; col++) {
			if (filtered_buffer[row * filtered_image.w + col] == true) {
				
				// Add a count to the number of green pixels
				*green_pixels += 1;

				// Determine which ray the pixel belongs to
				float angle = (float) atan2((double) row * kernel_size_h, (double) (col - filtered_image.w / 2) * kernel_size_w);
				uint8_t err_angle_min_idx = 0;
				float err_angle_min = 2.0 * M_PI;
				for (uint8_t i=0; i<7; i++) {
					float err_angle = fabsf(angle - ray_angles[i]);
					if (err_angle < err_angle_min) {
						err_angle_min_idx = i;
						err_angle_min = err_angle;
					}
				}

				// Add a score to the corresponding ray
				ray_scores[err_angle_min_idx] += 1.0f;
			}
		}
	}

	// Deallocate memory
  image_free(&filtered_image);

	// Go through ray scores, multiplying by corresponding weights and keep track of ray with highest score
	uint8_t best_heading_idx=0;
	float best_heading_score = 0.0f;
	for (uint8_t i=0; i<7; i++) {
		float score = ray_scores[i] * ray_weights[i];
		if (score > best_heading_score) {
			*safe_length = score;
			best_heading_score = score;
			best_heading_idx = i;
		}
	}

	// Assign remaining results
	*best_heading = ray_angles[best_heading_idx];
	*green_pixels = *green_pixels * kernel_size_w * kernel_size_h;
  VERBOSE_PRINT("GF: total pixels %d\n", *green_pixels);
}




// //
// // Created by lilly on 6-3-24.
// //

// #include "modules/green_follower/green_detector.h"
// #include "modules/computer_vision/cv.h"
// #include "modules/core/abi.h"
// #include "std.h"

// #include <stdio.h>
// #include <stdbool.h>
// #include <math.h>
// #include "pthread.h"

// #ifndef GREENFILTER_FPS
// #define GREENFILTER_FPS 0       ///< Default FPS (zero means run at camera fps)
// #endif
// PRINT_CONFIG_VAR(COLORFILTER_FPS)

// #define GREEN_DETECTOR_VERBOSE TRUE

// #define PRINT(string,...) fprintf(stderr, "[green_follower->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
// #if GREEN_DETECTOR_VERBOSE
// #define VERBOSE_PRINT PRINT
// #else
// #define VERBOSE_PRINT(...)
// #endif

// // Filter Settings
// uint8_t gd_lum_min = 60;
// uint8_t gd_lum_max = 130;
// uint8_t gd_cb_min = 75;
// uint8_t gd_cb_max = 110;
// uint8_t gd_cr_min = 120;
// uint8_t gd_cr_max = 140;

// static pthread_mutex_t mutex;

// struct heading_object_t {
//     float best_heading;
//     float safe_length;
//     uint32_t green_pixels;
//     bool updated;
// };
// struct heading_object_t global_heading_object;

// void apply_threshold(struct image_t *img, uint32_t *green_pixels,
//                      uint8_t lum_min, uint8_t lum_max,
//                      uint8_t cb_min, uint8_t cb_max,
//                      uint8_t cr_min, uint8_t cr_max);

// float get_radial(struct image_t *img, float angle, uint8_t radius);

// void get_direction(struct image_t *img, uint8_t resolution, float* best_heading, float* safe_length);

// /*
//  * object_detector
//  * @param img - input image to process
//  * @return img
//  */
// static struct image_t *green_heading_finder(struct image_t *img)
// {
//     uint8_t lum_min, lum_max;
//     uint8_t cb_min, cb_max;
//     uint8_t cr_min, cr_max;

//     float best_heading, safe_length;

//     uint32_t green_pixels;

//     lum_min = gd_lum_min;
//     lum_max = gd_lum_max;
//     cb_min = gd_cb_min;
//     cb_max = gd_cb_max;
//     cr_min = gd_cr_min;
//     cr_max = gd_cr_max;
//     uint8_t scan_resolution = 50;

//     // Filter the image so that all green pixels have a y value of 255 and all others a y value of 0
//     apply_threshold(img, &green_pixels, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
//     // Scan in radials from the centre bottom of the image to find the direction with the most green pixels
//     get_direction(img, scan_resolution, &best_heading, &safe_length);

//     pthread_mutex_lock(&mutex);
//     global_heading_object.best_heading = best_heading;
//     global_heading_object.safe_length = safe_length;
//     global_heading_object.green_pixels = green_pixels;
//     global_heading_object.updated = true;
//     pthread_mutex_unlock(&mutex);
//     return img;
// }

// struct image_t *green_heading_finder1(struct image_t *img, uint8_t camera_id);
// struct image_t *green_heading_finder1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
// {
//     return green_heading_finder(img);
// }

// void green_detector_init(void) {
//     memset(&global_heading_object, 0, sizeof(struct heading_object_t));
//     pthread_mutex_init(&mutex, NULL);

//     #ifdef GREEN_DETECTOR_LUM_MIN
//         gd_lum_min = GREEN_DETECTOR_LUM_MIN;
//         gd_lum_max = GREEN_DETECTOR_LUM_MAX;
//         gd_cb_min = GREEN_DETECTOR_CB_MIN;
//         gd_cb_max = GREEN_DETECTOR_CB_MAX;
//         gd_cr_min = GREEN_DETECTOR_CR_MIN;
//         gd_cr_max = GREEN_DETECTOR_CR_MAX;
//     #endif

//     cv_add_to_device(&GREENFILTER_CAMERA, green_heading_finder1, GREENFILTER_FPS, 0);
// }

// void green_detector_periodic(void) {
//     static struct heading_object_t local_heading_object;
//     pthread_mutex_lock(&mutex);
//     memcpy(&local_heading_object, &global_heading_object, sizeof(struct heading_object_t));
//     pthread_mutex_unlock(&mutex);

//     if(local_heading_object.updated){
//         AbiSendMsgGREEN_DETECTION(GREEN_DETECTION_ID, local_heading_object.best_heading, local_heading_object.safe_length, local_heading_object.green_pixels);
//         local_heading_object.updated = false;
//     }
// }


// void apply_threshold(struct image_t *img, uint32_t* green_pixels,
//                      uint8_t lum_min, uint8_t lum_max,
//                      uint8_t cb_min, uint8_t cb_max,
//                      uint8_t cr_min, uint8_t cr_max)
// {
//   uint8_t *buffer = img->buf;
//   uint32_t local_green_pixels = 0;

//   // Go through all the pixels
//   for (uint16_t y = 0; y < img->h; y++) {
//     for (uint16_t x = 0; x < img->w; x ++) {
//       // Check if the color is inside the specified values
//       uint8_t *yp, *up, *vp;
//       if (x % 2 == 0) {
//         // Even x
//         up = &buffer[y * 2 * img->w + 2 * x];      // U
//         yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y1
//         vp = &buffer[y * 2 * img->w + 2 * x + 2];  // V
//         //yp = &buffer[y * 2 * img->w + 2 * x + 3]; // Y2
//       } else {
//         // Uneven x
//         up = &buffer[y * 2 * img->w + 2 * x - 2];  // U
//         //yp = &buffer[y * 2 * img->w + 2 * x - 1]; // Y1
//         vp = &buffer[y * 2 * img->w + 2 * x];      // V
//         yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y2
//       }
//       if ( (*yp >= lum_min) && (*yp <= lum_max) &&
//            (*up >= cb_min ) && (*up <= cb_max ) &&
//            (*vp >= cr_min ) && (*vp <= cr_max )) {
//             local_green_pixels++;
//             *yp = 255;  // make pixel white
//         }
//       else {
//         *yp = 0; // make pixel black
//       }
//     }
//   }
//   *green_pixels = local_green_pixels;
// }

// float get_radial(struct image_t *img, float angle, uint8_t radius) {
//     uint8_t *buffer = img->buf;

//     uint32_t sum = 0;
//     uint16_t x, y;

//     for (double i = 0; i < radius; i++) {
//         y = (uint16_t)((double)img->h - i * sin(angle));
//         x = (uint16_t)((double)img->w / 2 + i * cos(angle));
//         sum = sum + buffer[y * 2 * img->w + 2 * x + 1];
//     }

//     return (float)sum * (sin(angle) + 0.2) ;
// }

// void get_direction(struct image_t *img, uint8_t resolution, float* best_heading, float* safe_length) {

//     float step_size = M_PI / (float)resolution;
//     *best_heading = 0;
//     *safe_length = 0;

//     for (float angle = 0; angle < M_PI; angle += step_size) {
//         float radial = get_radial(img, angle, img->w / 2);

//         if (radial >= *safe_length) {
//             *best_heading = angle;
//             *safe_length = radial;
//         }
//     }

//     *best_heading = M_PI/2 - *best_heading;
// }
