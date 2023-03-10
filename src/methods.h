#ifndef METHODS_H
#define METHODS_H
enum FocusMethod{ONE_DISTANCE, BRUTE_FORCE, RANDOM, HIERARCHY, DESCENT, PYRAMID};
enum ColorDistance{RGB, YUV, Y, YUVw};
enum ScanMetric{VARIANCE, RANGE, IQR, MAD};
enum AddressMode{WRAP, CLAMP, MIRROR, BORDER, BLEND};
enum ClosestFrames{TOP_LEFT=0, TOP_RIGHT=1, BOTTOM_LEFT=2, BOTTOM_RIGHT=3};
enum IntConstantIDs{IMG_RES_X=0, IMG_RES_Y, COLS, ROWS, GRID_SIZE, DISTANCE_ORDER, SCAN_METRIC, FOCUS_METHOD, CLOSEST_VIEWS, BLOCK_SAMPLING, YUV_DISTANCE, CLOCK_SEED, BLEND_ADDRESS_MODE, NO_MAP, INT_CONSTANTS_COUNT};
enum FloatConstantIDs{SCAN_RANGE, SPACE, DESCENT_START_STEP, FOCUS_METHOD_PARAMETER, FLOAT_CONSTANTS_COUNT};
constexpr int DESCENT_START_POINTS{6};
constexpr int HIERARCHY_DIVISIONS{10};
enum DataPointersIDs{SURFACES, TEXTURES, SECONDARY_TEXTURES, MIP_TEXTURES, WEIGHTS, CLOSEST_WEIGHTS, CLOSEST_COORDS, POINTERS_COUNT};
enum FileNames {FOCUS_MAP=0, RENDER_IMAGE=1};
constexpr int CHANNEL_COUNT{4};
#endif
