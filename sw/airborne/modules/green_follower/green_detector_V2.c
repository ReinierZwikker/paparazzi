//
// Copied from lilly's v1 and modified
//

#include "modules/green_follower/green_detector.h"
#include "pthread.h"
#include <math.h>

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
uint8_t gd_lum_min = 80;
uint8_t gd_lum_max = 255;
uint8_t gd_cb_min = 30;
uint8_t gd_cb_max = 89;
uint8_t gd_cr_min = 90;
uint8_t gd_cr_max = 145;
const uint8_t kernel_size_w = 10;	// Note: Needs to be an integer divider of the input image pixel width 
const uint8_t kernel_size_h = 20; // Note: Needs to be an integer divider of the input image pixel height
// NOTE [Aaron]: If we make it such that the number of pixels in the reduced image (after the filter) is divisible by 8, 
// we could do bitwise boolean allocation to save on memory, but waaay overkill for now

const float ray_angles[] = {EIGEN_PI/6.0f, 
														EIGEN_PI/6.0f + 			 EIGEN_PI/9.0f, 
														EIGEN_PI/6.0f + 2.0f * EIGEN_PI/9.0f, 
														EIGEN_PI/6.0f + 3.0f * EIGEN_PI/9.0f, 
														EIGEN_PI/6.0f + 4.0f * EIGEN_PI/9.0f, 
														EIGEN_PI/6.0f + 5.0f * EIGEN_PI/9.0f, 
														EIGEN_PI/6.0f + 6.0f * EIGEN_PI/9.0f};


static pthread_mutex_t mutex;


struct filtered_image_T {
	uint16_t w;
	uint16_t h;
	uint16_t center_x;
	uint16_t center_y;
	uint32_t buffer_size;
	bool buffer[];
}

static void init_filtered_image(struct filtered_image_T* filtered_image, width, height) {
	filtered_image->w = width;
	filtered_image->h = height;
	filtered_image->center_x = filtered_image->w / 2
	filtered_image->center_y = filtered_image->h / 2
	filtered_image->buffer_size = filtered_image->w * filtered_image->h;

#if __GLIBC__ > 2 || (__GLIBC__ >= 2 && __GLIBC_MINOR__ >= 16)
	// NOTE [Aaron]: Actually no clue how this first line works but should help I guess, was taken from image.c
  // aligned memory slightly speeds up any later copies
  filtered_image->buffer = aligned_alloc(CACHE_LINE_LENGTH, filtered_image->buffer_size + (CACHE_LINE_LENGTH - filtered_image->buffer_size % CACHE_LINE_LENGTH) % CACHE_LINE_LENGTH);
#else
  filtered_image->buffer = malloc(filtered_image->buffer_size);
#endif
}

static void free_filtered_image(strcut filtered_image_T* filtered_image) {
	if (filtered_image->buffer != NULL) {
    free(filtered_image->buffer);
    filtered_image->buffer = NULL;
  }
}


static void green_filter(struct image_t* original_image, struct filtered_image_T* filtered_image) {

	uint8_t *buffer = original_image->buf;

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
						yuv_u += buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)];      // U
						yuv_y += buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1];  // Y1
						yuv_v += buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 2];  // V
					} 
					else {
						// Uneven col nr
						yuv_u += buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) - 2];  // U
						yuv_v += buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)];      // V
						yuv_y += buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1];  // Y2
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

				filtered_image->buffer[row * filtered_image_w + col] = true;
			}
			else {
				filtered_image->buffer[row * filtered_image_w + col] = false;
			}
		}
	}
}


void get_direction(struct image_t* original_image, float* best_heading) {

	// Array where ray scores will be added
	float ray_scores[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	// Array which will contain the filtered image
	struct filtered_image_T filtered_image;
	init_filtered_image(&filtered_image);

	// Apply filter to original image
	green_filter(original_image, &filtered_image)

	// Go through each pixel in the filtered image
	// and determine which ray it belongs to
	for (uint16_t row=0; row<filtered_image.h; row++) {
		for (uint16_t col=0; col<filtered_image.w; col++) {
			if (filtered_image.buffer[row * filtered_image.w + col] == true) {
				float angle = (float) atan2((double) row * kernel_size_h, (double) (col - filtered_image.center_x) * kernel_size_w);

				uint8_t err_angle_min_idx = 0;
				float err_angle_min = 2.0 * EIGEN_PI
				for (uint8_t i=0; i<7; i++) {
					float err_angle = abs(angle - ray_angles[i]);
					if (err_angle < err_angle_min) {
						err_angle_min_idx = i;
						err_angle_min = err_angle;
					}
				}

				ray_scores[err_angle_min_idx] += 1.0f;

			}
		}
	}

	// Deallocate memory
	free_filtered_image(&filtered_image);

	// Go through ray scores, multiplying by corresponding weights and keep track of ray with highest score
	uint8_t best_heading_idx=0;
	float best_heading_score = 0.0f;
	for (uint8_t i=0; i<7; i++) {
		if (ray_scores[i] *  > best_heading_score) {

		}
	}

	
}
