#pragma once

#include "eel.h"

void check_state_for_nan(const char* context, const struct Config *config, struct InferState *state);

void check_weights_for_nan(const char *context, const struct Config *config, const struct Weights *weights);

void print_head_tail(const float *arr, int n, int h);