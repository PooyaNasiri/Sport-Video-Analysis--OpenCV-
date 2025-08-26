/********************************************************************************
  Project: Sport Video Analisis
  Author: Pooya Nasiri (Student ID: 2071437)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
// eval_iou.cpp
// Usage: ./eval_iou <ours.csv> <yolo.csv> [iou_thr=0.5] [ours_offset=0] [yolo_offset=0]
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct Box
{
    double x1, y1, x2, y2;
};

static inline void strip_cr(std::string &s)
{
    if (!s.empty() && s.back() == '\r')
        s.pop_back();
}
static inline void strip_bom(std::string &s)
{
    const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
    if (s.size() >= 3 &&
        (unsigned char)s[0] == bom[0] &&
        (unsigned char)s[1] == bom[1] &&
        (unsigned char)s[2] == bom[2])
    {
        s.erase(0, 3);
    }
}
static inline bool is_header_like(const std::vector<std::string> &t)
{
    if (t.size() < 5)
        return true;
    // crude check: any token has alpha
    for (const auto &x : t)
    {
        for (char c : x)
            if (std::isalpha((unsigned char)c))
                return true;
    }
    return false;
}
static inline std::vector<std::string> split_csv(const std::string &line)
{
    // simple CSV split (no quoted fields in our use-case)
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        // trim surrounding spaces
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a == std::string::npos)
            out.emplace_back("");
        else
            out.emplace_back(tok.substr(a, b - a + 1));
    }
    return out;
}

static double iou(const Box &a, const Box &b)
{
    const double x1 = std::max(a.x1, b.x1);
    const double y1 = std::max(a.y1, b.y1);
    const double x2 = std::min(a.x2, b.x2);
    const double y2 = std::min(a.y2, b.y2);
    const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double a1 = std::max(0.0, a.x2 - a.x1) * std::max(0.0, a.y2 - a.y1);
    const double a2 = std::max(0.0, b.x2 - b.x1) * std::max(0.0, b.y2 - b.y1);
    const double uni = a1 + a2 - inter;
    return uni > 0 ? inter / uni : 0.0;
}

static void load_csv_5cols(const std::string &path,
                           std::map<int, std::vector<Box>> &by_frame,
                           int frame_offset)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open " + path);
    }

    std::string line;
    bool first = true;
    while (std::getline(f, line))
    {
        strip_cr(line);
        if (line.empty())
            continue;

        if (first)
        {
            strip_bom(line);
            first = false;
        }
        if (line.size() && line[0] == '#')
            continue;

        auto t = split_csv(line);
        if (t.size() < 5)
            continue;
        if (is_header_like(t))
            continue; // skip header-like lines anywhere

        try
        {
            int frame = std::stoi(t[0]) + frame_offset;
            Box b{std::stod(t[1]), std::stod(t[2]), std::stod(t[3]), std::stod(t[4])};
            by_frame[frame].push_back(b);
        }
        catch (...)
        {
            // ignore malformed rows
            continue;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <ours.csv> <yolo.csv> [iou_thr=0.5] [ours_offset=0] [yolo_offset=0]\n";
        return 1;
    }
    const std::string ours_path = argv[1];
    const std::string yolo_path = argv[2];
    const double thr = (argc >= 4) ? std::stod(argv[3]) : 0.5;
    const int off_ours = (argc >= 5) ? std::stoi(argv[4]) : 0;
    const int off_yolo = (argc >= 6) ? std::stoi(argv[5]) : 0;

    std::map<int, std::vector<Box>> ours, yolo;
    try
    {
        load_csv_5cols(ours_path, ours, off_ours);
        load_csv_5cols(yolo_path, yolo, off_yolo);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Load error: " << e.what() << "\n";
        return 2;
    }

    long TP = 0, FP = 0, FN = 0;
    std::vector<double> matched_ious;

    std::set<int> frames;
    for (auto &kv : ours)
        frames.insert(kv.first);
    for (auto &kv : yolo)
        frames.insert(kv.first);

    for (int f : frames)
    {
        auto &P = ours[f]; // predictions
        auto &G = yolo[f]; // ground truth (YOLO)
        std::vector<char> used(G.size(), 0);

        for (const auto &pb : P)
        {
            double best = 0.0;
            int best_j = -1;
            for (int j = 0; j < (int)G.size(); ++j)
            {
                if (used[j])
                    continue;
                double v = iou(pb, G[j]);
                if (v > best)
                {
                    best = v;
                    best_j = j;
                }
            }
            if (best >= thr)
            {
                TP++;
                used[best_j] = 1;
                matched_ious.push_back(best);
            }
            else
            {
                FP++;
            }
        }
        for (int j = 0; j < (int)G.size(); ++j)
            if (!used[j])
                FN++;
    }

    const double precision = (TP + FP) ? double(TP) / (TP + FP) : 0.0;
    const double recall = (TP + FN) ? double(TP) / (TP + FN) : 0.0;
    const double f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;
    const double miou = matched_ious.empty()
                            ? 0.0
                            : std::accumulate(matched_ious.begin(), matched_ious.end(), 0.0) / matched_ious.size();

    std::cout << "TP=" << TP << " FP=" << FP << " FN=" << FN << "\n";
    std::cout << std::fixed << std::setprecision(3)
              << "Precision=" << precision
              << " Recall=" << recall
              << " F1=" << f1
              << " mIoU=" << miou << "\n";

    return 0;
}
