//for profiler and logger preformance
#ifndef PIPE_UTILS_H
#define PIPE_UTILS_H

#include<iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include "NvInfer.h"
#include <cassert>


#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)




class Logger : public nvinfer1::ILogger
{
    public:
        Logger(bool verbose):mVerbose{verbose}{
        }

        virtual void log(Severity severity, const char* msg) override
        {
            int em = (int)severity;
            if(mVerbose)
                std::cerr << em << ": " << msg << std::endl;
        }
    private:
        bool mVerbose;
};

// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
{
const int MAX_DEPTH{10};
bool found{false};
std::string filepath;

for (auto& dir : directories)
{
    if (!dir.empty() && dir.back() != '/')
    {
#ifdef _MSC_VER
        filepath = dir + "\\" + filepathSuffix;
#else
        filepath = dir + "/" + filepathSuffix;
#endif
    }
    else
        filepath = dir + filepathSuffix;

    for (int i = 0; i < MAX_DEPTH && !found; i++)
    {
        std::ifstream checkFile(filepath);
        found = checkFile.is_open();
        if (found)
            break;
        filepath = "../" + filepath; // Try again in parent dir
    }

    if (found)
    {
        break;
    }

    filepath.clear();
}

if (filepath.empty())
{
    std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
        [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
    std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << directoryList << std::endl;
    std::cout << "&&&& FAILED" << std::endl;
    exit(EXIT_FAILURE);
}
return filepath;
}

inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

#endif