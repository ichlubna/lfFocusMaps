This repository contains tools for light field rendering using the focus map. The implementation was used in *Lightweight All-Focused Light Field Rendering* paper. Visit the [research page](https://www.fit.vutbr.cz/~ichlubna/lf) for more details and the dataset.

# Content
*src* - contains the source codes for the CUDA-based focus map generator and novel view renderer main application (use *CMakeLists.txt* to build and *-h* argument for correct usage instructions)

*Blender* - contains a simplified light field renderer addon and optimal capturing addon for synthetic scenes (this addon was used to produce the dataset used with the main application)

*scripts* - contains several scripts for measurements of the results produced by the main application

# Installation of Blender addons
Pack the *.py* and *.blend* files in the addon directory into a zip file. Open Blender->Edit->Preferences->Add-ons->Install and select the zip file. The GUI should appear in the side panel in the viewport. Make sure to use the Viewport Shading view set to Rendered to see the final results.

# Citation
Please cite our work if you find this repository useful:
```
@ARTICLE{Chlubna2024,
   author = "Tom\'{a}\v{s} Chlubna and Tom\'{a}\v{s} Milet and Pavel Zem\v{c}\'{i}k",
   title = "Lightweight All-Focused Light Field Rendering",
   pages = "7--8",
   journal = "Computer Vision and Image Understanding",
   volume = 244,
   number = 7,
   year = 2024,
   ISSN = "1077-3142",
   doi = "10.1016/j.cviu.2024.104031",
   language = "english",
   url = "https://www.fit.vut.cz/research/publication/13204"
}
```
