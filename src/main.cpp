#include "interpolator.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string path = static_cast<std::string>(args["-i"]);
    std::string coordinates = static_cast<std::string>(args["-c"]);
    std::string outputPath = static_cast<std::string>(args["-o"]);
    std::string method = static_cast<std::string>(args["-m"]);

    std::string helpText{ "Usage:\n"
                          "Example: lfInterpolator -i /MyAmazingMachine/thoseImages -t 0.0,0.0,1.0,1.0  -o ./outputs\n"
                          "-i - folder with lf grid images - named as column_row.extension, e.g. 01_12.jpg\n"
                          "-c - camera position in normalized coordinates of the grid in format: x_y, e.g., 0.5_0.1\n"
                          "-o - output path\n"
                          "-m - method: BF - brute force\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;

    if(!args["-i"] || !args["-c"] || !args["-o"])
    {
        std::cerr << "Missing required parameters. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        Interpolator interpolator(path);
        interpolator.interpolate(outputPath, coordinates, method);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
