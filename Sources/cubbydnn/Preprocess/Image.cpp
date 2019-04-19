
#include "cubbydnn/Preprocess/Image.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace CubbyDNN
{
    Image::Image()
    {
        image_defined = false;
    }

    Image Image::Load(std::string filename, int load_color_type)
    {
        Image result;
        size_t pos_after_dot = filename.find_last_of('.') + 1;
        if (pos_after_dot == filename.length())
            throw std::runtime_error("Filename Invalid");
        std::string extension =
            filename.substr(pos_after_dot, filename.length() - pos_after_dot);
        std::basic_ifstream<unsigned char> image_file(filename);

        if (extension == "bmp")
        {
            result = load_bmp(image_file);
        }
        switch (load_color_type)
        {
            case COLOR_RGB:
                // TODO
                break;
        }

        image_file.close();
        return result;
    }

    Image Image::load_bmp(std::basic_ifstream<unsigned char>& f)
    {
        Image result;

        unsigned char info[54];
        f.read(info, 54);
        int width = *reinterpret_cast<int*>(&info[18]);
        int height = *reinterpret_cast<int*>(&info[22]);

        result.w = width;
        result.h = height;

        int row_padded = (width * 3 + 3) & (~3);
        unsigned char* data = new unsigned char[row_padded];

        result.image = new unsigned char**[height];
        for (int i = 0; i < height; i++)
        {
            result.image[i] = new unsigned char*[width];
            f.read(data, row_padded);
            for (int j = 0; j < width; j++)
            {
                result.image[i][j] = new unsigned char[3];
                // BGR -> RGB
                result.image[i][j][0] = data[3*j + 2];
                result.image[i][j][1] = data[3*j + 1];
                result.image[i][j][2] = data[3*j];
            }
        }
        result.image_defined = true;
        delete[] data;
        return result;
    }

    Image Image::Load(std::string filename)
    {
        return Load(filename, Image::COLOR_RGB);
    }

    void Image::PrintImageTest(std::ostream& output)
    {
        output << "image size ( " << w << ", " << h << " )" << std::endl;
        for (int i = 0; i < h; i++)
        {
            output << "[ ";
            for (int j = 0; j < w; j++)
            {
                output << "( ";
                for (int z = 0; z < 3; z++)
                {
                    output << static_cast<int>(image[i][j][z])
                              << ((z != 2) ? ", " : "");
                }
                output << " )";
            }
            output << "]" << std::endl;
        }
    }
	
}  // namespace CubbyDNN