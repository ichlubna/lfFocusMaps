#include "interpolator.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string helpText{ "Usage:\n"
                          "Example: lfInterpolator -i /MyAmazingMachine/thoseImages -t 0.0,0.0,1.0,1.0  -o ./outputs\n"
                          "-i - folder with lf grid images - named as column_row.extension, e.g. 01_12.jpg\n"
                          "-c - camera position in normalized coordinates of the grid in format: x_y, e.g., 0.5_0.1\n"
                          "-o - output path\n"
                          "-m - method:\n"
                          "     OD (one distance) - parameter: focusing distance in pixels\n"
                          "     BF (brute force) - parameter: number of scanning steps\n"
                          "-p - method parameter\n"
                          "-b - block sampling - uses also neighboring pixels for matching\n"
                          "-f - use faster variant with only four closest views\n"
                          "-r - scanning range in pixels - the maximum disparity between input images, default is half of image width\n"
                          "-d - order of the distance function, e.g., 2 => distance = distance^2, default is 1"
                          "-t - number of kernel runs for performance measurement - default is 1\n"
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
        Interpolator interpolator(static_cast<std::string>(args["-i"]));
        Interpolator::InterpolationParams params;
        params
        .setMethod(static_cast<std::string>(args["-m"]))
        ->setMethodParameter(static_cast<float>(args["-p"]))
        ->setCoordinates(static_cast<std::string>(args["-c"]))
        ->setScanRange(static_cast<int>(args["-r"]))
        ->setOutputPath(static_cast<std::string>(args["-o"]))
        ->setRuns(std::stoi(static_cast<std::string>(args["-t"])))
        ->setDistanceOrder(static_cast<int>(args["-d"]))
        ->setBlockSampling(static_cast<bool>(args["-b"]))
        ->setClosestViews(static_cast<bool>(args["-f"]));
        interpolator.interpolate(params);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
