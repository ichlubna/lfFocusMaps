#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <cstring>
#include "libs/loadingBar/loadingbar.hpp"
#include "lfLoader.h"

const std::set<std::filesystem::path> LfLoader::listPath(std::string path) const
{
    if(!std::filesystem::exists(path))
        throw std::runtime_error("The path "+path+" does not exist!");
    if(!std::filesystem::is_directory(path))
        throw std::runtime_error("The path "+path+" does not lead to a directory!");

    auto dir = std::filesystem::directory_iterator(path);
    std::set<std::filesystem::path> sorted;
    for(const auto &file : dir)
        sorted.insert(file.path().filename());
    return sorted;
}

glm::ivec2 LfLoader::parseFilename(std::string name) const
{
    auto delimiterPos = name.find('_');
    if(delimiterPos == std::string::npos)
        throw std::runtime_error("File "+name+" is not named properly as column_row.extension!");
    int extensionPos = name.find('.');
    auto row = name.substr(0, delimiterPos);
    auto col = name.substr(delimiterPos + 1, extensionPos - delimiterPos - 1);
    return {stoi(row), stoi(col)};
}

void LfLoader::loadImage(std::string path, glm::uvec2 coords)
{
    constexpr int RGBA_CHANNELS{4};
    uint8_t *pixels = stbi_load(path.c_str(), &resolution.x, &resolution.y, &resolution.z, STBI_rgb_alpha);
    if(pixels == nullptr)
        throw std::runtime_error("Cannot load image " + path);
    resolution.z = RGBA_CHANNELS;
    size_t size = resolution.x*resolution.y*resolution.z;
    grid[coords.x][coords.y] = std::vector<uint8_t>(pixels, pixels+size);
}

void LfLoader::initGrid(glm::uvec2 inColsRows)
{
    colsRows = inColsRows+glm::uvec2(1);
    grid.resize(colsRows.x);
    for(auto &row : grid)
        row.resize(colsRows.y);
}

void LfLoader::loadData(std::string path)
{  
    auto files = listPath(path);
    if(files.empty())
        throw std::runtime_error("The input directory is empty!");
    initGrid(parseFilename(*files.rbegin()));

    std::cout << "Loading images..." << std::endl;
    LoadingBar bar(files.size());
    for(auto const &file : files)
    {
        auto coords = parseFilename(file);
        loadImage(path/file, coords); 
        bar.add(); 
    }
}
