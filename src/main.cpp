#include "interpolator.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string helpText{ "Usage:\n"
                          "-i - folder with lf grid images - named as row_column.extension, e.g.: 01_12.jpg\n"
                          "-l - uses secondary folder with same filenames for focusing, e.g.: -i my/folder => my contains folder and folder_sec (flag) \n"
                          "-c - camera position in normalized coordinates of the grid in format: x_y, e.g., 0.5_0.1\n"
                          "-o - output path\n"
                          "-m - method:\n"
                          "     OD (one distance) - parameter: focusing distance in pixels\n"
                          "     BF (brute force) - parameter: number of scanning steps (default)\n"
                          "     BFET (brute force with early termination) - the same as BF but ends if the optimum does not change\n"
                          "     VS (variable scan) - parameter: initial step size - the number of divisions of the range\n"
                          "     VSET (variable scan with early termination)\n"
                          "     TD (top down)\n"
                          "     RAND (random sampling)\n"
                          "     HIER (hierarchichal sampling)\n"
                          "     DESC (3-way descent)\n"
                          "     PYR (pyramid approach) - The first phase samples downscaled images that need to be stored in folder\n"
                          "     named the same as input folder with same filenames\n"
                          "     e.g.: -i my/folder => my contains folder and folder_down\n"
                          "-p - method parameter\n"
                          "-e - scan metric to evaluate the color dispersion:\n"
                          "     VAR - variance (default)\n"
                          "     ERAN - elementwise min/max distances between the colors\n"
                          "     RAN - range of the colors in relation to the zero distance\n"
                          "     MAD - approximation of mean absolute difference\n"
                          "-s   scan space - density of the sampling by exponential function y=x^s, default s=1\n"
                          "-y   color distance metric:\n"  
                          "     RGB - Chebyshev RGB distance (default)\n"
                          "     Y - difference between Y values (intensity without color)\n"
                          "     YUV - Chebyshev YUV\n"
                          "     YUVw - Weighted Chebyshev YUV - more focus on colors\n"
                          "-b - block sampling - uses also neighboring pixels for matching and takes float number of pixels > 0 (0 means that no block sampling will be used and is default)\n"
                          "-f - use faster variant with only four closest views (flag)\n"
                          "-r - normalized scanning range - the maximum and minimum disparity between input images, default is from zero to half of image width - 0.0_0.5\n"
                          "-d - order of the distance function, e.g., 2 => distance = distance^2, default is 1\n"
                          "-t - number of kernel runs for performance measurement - default is 1\n"
                          "-a - address mode - what to sample outside the images: WRAP, CLAMP, MIRROR, BORDER, BLEND, ALTER - default is CLAMP\n"
                          "-g - aspect ratio of the capturing camera spacing in horizontal / vertical axis, 1 means that the cameras are spaced equally in x and y\n"
                          "     if this option is not used or set to 0, the lf grid spacing is expected to be in the same ratio as resolution \n"
                          "-x - simulates post-process depth of field: accepts value focusDistance_width_maxBlur - same units as -r\n"
                          "-w - simulates post-process mist: accepts value start_end_hexaColor(#BBGGRR) - first two (far start and near end) are same units as -r\n"
                          "-z - focus map filter in post-process, the following methods can be stacked as FIRST_SECOND_THIRD_...:\n"
                          "     NONE (default)\n"
                          "     MED - median\n"
                          "     SNN - symmetric nearest neighbor\n"
                          "     KUW - Kuwahara filter\n"
                          "Example:\n"
                          "lfFocusMaps -i /MyAmazingMachine/theImages -o ./outputs -c 0.5_0.5 -m BF -p 32 -g 1 -z MED -x 0.5_0.1_0.2\n"
                          "The output folder contains:\n"
                          "focusMap.hdr - the computed focus map in HDR format as 32bit float per pixel\n"
                          "focusMapPost.hdr - the filtered focus map used in the post processing\n"
                          "renderImage.png - the novel interpolated view at position defined by -c\n"
                          "renderImagePost.png - the novel view with post-processing effects (e.g. -x or -n)\n"
                          "renderImagePostFiltered.png - the novel view rendered with the filtered map\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;
    
    if(!args["-i"] || !args["-c"] || !args["-o"] || !args["-m"])
    {
        std::cerr << "Missing required parameters. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }
    
    try
    {
        bool useMips = static_cast<std::string>(args["-m"]) == "PYR";
        bool yuv = static_cast<std::string>(args["-y"])[0] == 'Y';
        Interpolator interpolator(static_cast<std::string>(args["-i"]), static_cast<std::string>(args["-a"]), static_cast<bool>(args["-l"]), useMips, yuv, args["-g"]);
        Interpolator::InterpolationParams params;
        params
        .setMethod(static_cast<std::string>(args["-m"]))
        ->setMetric(static_cast<std::string>(args["-e"]))
        ->setSpace(static_cast<float>(args["-s"]))
        ->setMethodParameter(static_cast<float>(args["-p"]))
        ->setCoordinates(static_cast<std::string>(args["-c"]))
        ->setScanRange(static_cast<std::string>(args["-r"]))
        ->setOutputPath(static_cast<std::string>(args["-o"]))
        ->setRuns(std::stoi(static_cast<std::string>(args["-t"])))
        ->setDistanceOrder(static_cast<int>(args["-d"]))
        ->setBlockSampling(static_cast<float>(args["-b"]))
        ->setClosestViews(static_cast<bool>(args["-f"]))
        ->setColorDistance(static_cast<std::string>(args["-y"]))
        ->setMapFilter(static_cast<std::string>(args["-z"]))
        ->setMist(static_cast<std::string>(args["-w"]))
        ->setDof(static_cast<std::string>(args["-x"]));
        interpolator.interpolate(params);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
