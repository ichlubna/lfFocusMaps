#include "interpolator.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string helpText{ "Usage:\n"
                          "Example: lfInterpolator -i /MyAmazingMachine/thoseImages -t 0.0,0.0,1.0,1.0  -o ./outputs\n"
                          "-i - folder with lf grid images - named as column_row.extension, e.g.: 01_12.jpg\n"
                          "-l - uses secondary folder with same filenames for focusing, e.g.: -i my/folder => my contains folder and folder_sec \n"
                          "-c - camera position in normalized coordinates of the grid in format: x_y, e.g., 0.5_0.1\n"
                          "-o - output path\n"
                          "-m - method:\n"
                          "     OD (one distance) - parameter: focusing distance in pixels\n"
                          "     BF (brute force) - parameter: number of scanning steps (default)\n"
                          "     RAND (random sampling) - parameter: number of scanning samples\n"
                          "     HIER (hierarchichal sampling) - parameter: 1 randomized, 0 uniform \n"
                          "     DESC (3-way descent) - parameter: 1 randomized, 0 uniform\n"
                          "     PYR (pyramid approach) - parameter: 1 randomized, 0 uniform\n"
                          "     The first phase samples downscaled images that need to be stored in folder\n"
                          "     named the same as input folder with same filenames\n"
                          "     e.g.: -i my/folder => my contains folder and folder_down\n"
                          "-p - method parameter\n"
                          "-e - scan metric to evaluate the color dispersion:\n"
                          "     VAR - variance (default)\n"
                          "     RANGE - elementwise min/max distances between the colors\n"
                          "     IQR - inter-quartile range\n"
                          "     MAD - approximation of mean absolute difference\n"
                          "-s   scan space - density of the sampling by exponential function y=x^s, default s=1\n"
                          "-y   use weighted YUV color distance\n"
                          "-b - block sampling - uses also neighboring pixels for matching\n"
                          "-f - use faster variant with only four closest views\n"
                          "-r - normalized scanning range - the maximum disparity between input images, default is half of image width - 0.5\n"
                          "-d - order of the distance function, e.g., 2 => distance = distance^2, default is 1"
                          "-t - number of kernel runs for performance measurement - default is 1\n"
                          "-a - address mode - what to sample outside the images: WRAP, CLAMP, MIRROR, BORDER, BLEND - default is CLAMP\n"
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
        Interpolator interpolator(static_cast<std::string>(args["-i"]), static_cast<std::string>(args["-a"]));
        Interpolator::InterpolationParams params;
        params
        .setMethod(static_cast<std::string>(args["-m"]))
        ->setMetric(static_cast<std::string>(args["-e"]))
        ->setSpace(static_cast<float>(args["-s"]))
        ->setMethodParameter(static_cast<float>(args["-p"]))
        ->setCoordinates(static_cast<std::string>(args["-c"]))
        ->setScanRange(static_cast<float>(args["-r"]))
        ->setOutputPath(static_cast<std::string>(args["-o"]))
        ->setRuns(std::stoi(static_cast<std::string>(args["-t"])))
        ->setDistanceOrder(static_cast<int>(args["-d"]))
        ->setBlockSampling(static_cast<bool>(args["-b"]))
        ->setClosestViews(static_cast<bool>(args["-f"]))
        ->setYUVDistance(static_cast<bool>(args["-y"]))
        ->setSecondaryFocus(static_cast<bool>(args["-l"]));
        interpolator.interpolate(params);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
