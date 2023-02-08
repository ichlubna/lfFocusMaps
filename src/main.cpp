#include "interpolator.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string path = static_cast<std::string>(args["-i"]);
    std::string coordinates = static_cast<std::string>(args["-c"]);
    std::string outputPath = static_cast<std::string>(args["-o"]);
    std::string method = static_cast<std::string>(args["-m"]);
    float methodParameter = static_cast<float>(args["-p"]);
    bool closestViews = static_cast<bool>(args["-f"]);
    int range = static_cast<int>(args["-r"]);
    int runs = std::stoi(static_cast<std::string>(args["-t"]));

    std::string helpText{ "Usage:\n"
                          "Example: lfInterpolator -i /MyAmazingMachine/thoseImages -t 0.0,0.0,1.0,1.0  -o ./outputs\n"
                          "-i - folder with lf grid images - named as column_row.extension, e.g. 01_12.jpg\n"
                          "-c - camera position in normalized coordinates of the grid in format: x_y, e.g., 0.5_0.1\n"
                          "-o - output path\n"
                          "-m - method:\n"
                          "     OD (one distance) - parameter: focusing distance in pixels\n"
                          "     BF (brute force) - parameter: number of scanning steps\n"
                          "-p - method parameter\n"
                          "-f - use faster variant with only four closest views\n"
                          "-r - scanning range in pixels - the maximum disparity between input images (default is half of image width)\n"
                          "-t - number of kernel runs for performance measurement - default is one\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;

    if(!args["-i"] || !args["-c"] || !args["-o"] || !args["-m"])
    {
        std::cerr << "Missing required parameters. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }
    
    if(runs == 0)
        runs = 1;
    else if(runs < 0)
    {
        std::cerr << "Number of kernel runs cannot be negative. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        Interpolator interpolator(path);
        interpolator.interpolate(outputPath, coordinates, method, methodParameter, closestViews, range, runs);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
