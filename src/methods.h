#ifndef METHODS_H
#define METHODS_H
enum FocusMethod{ONE_DISTANCE, BRUTE_FORCE, BRUTE_FORCE_EARLY, TOP_DOWN, RANDOM, HIERARCHY, DESCENT, PYRAMID};
enum ColorDistance{RGB, YUV, Y, YUVw};
enum ScanMetric{VARIANCE, ELEMENT_RANGE, RANGE, MAD};
enum AddressMode{WRAP, CLAMP, MIRROR, BORDER, BLEND, ALTER};
enum MapFilter{NONE, MEDIAN, SNN};
enum ClosestFrames{TOP_LEFT=0, TOP_RIGHT=1, BOTTOM_LEFT=2, BOTTOM_RIGHT=3};
enum IntConstantIDs{IMG_RES_X=0, IMG_RES_Y, COLS, ROWS, GRID_SIZE, DISTANCE_ORDER, SCAN_METRIC, FOCUS_METHOD, CLOSEST_VIEWS, BLOCK_SAMPLING, YUV_DISTANCE, CLOCK_SEED, BLEND_ADDRESS_MODE, INT_CONSTANTS_COUNT};
enum FloatConstantIDs{SCAN_RANGE, SPACE, DESCENT_START_STEP, FOCUS_METHOD_PARAMETER, PYRAMID_BROAD_STEP, PYRAMID_NARROW_STEP, PX_SIZE_X, PX_SIZE_Y, DOF_DISTANCE, DOF_WIDTH, DOF_MAX, MIST_START, MIST_END, MIST_COLOR, FLOAT_CONSTANTS_COUNT};
constexpr int DESCENT_START_POINTS{6};
constexpr int HIERARCHY_DIVISIONS{10};
constexpr int PYRAMID_DIVISIONS_BROAD{10};
constexpr int PYRAMID_DIVISIONS_NARROW{5};
constexpr int BLOCK_OFFSET_COUNT{5};
enum DataPointersIDs{SURFACES, TEXTURES, SECONDARY_TEXTURES, MIP_TEXTURES, POINTERS_COUNT};
enum FileNames {FOCUS_MAP=0, FOCUS_MAP_POST, RENDER_IMAGE, RENDER_IMAGE_POST, RENDER_IMAGE_POST_MAP_FILTER, OUTPUT_COUNT};
constexpr int CHANNEL_COUNT{4};
#endif
