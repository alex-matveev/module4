#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

// Структура для хранения данных модели
struct ModelData {
    std::vector<std::vector<std::vector<double>>> depth;
    std::vector<std::vector<std::vector<double>>> velocity;
    std::vector<std::vector<double>> relief;
};

// Чтение данных модели из файлов
ModelData readModelData(const std::string& depthFile,
    const std::string& velocityFile,
    const std::string& reliefFile) {
    ModelData data;
    std::ifstream depthStream(depthFile);
    std::ifstream velocityStream(velocityFile);
    std::ifstream reliefStream(reliefFile);

    if (!depthStream.is_open() || !velocityStream.is_open() ||
        !reliefStream.is_open()) {
        std::cerr << "Не удалось открыть один из файлов!" << std::endl;
        return data;
    }

    // Чтение depth.txt
    int numSlices, numRows, numCols;
    depthStream >> numSlices >> numRows >> numCols;
    data.depth.resize(numSlices, std::vector<std::vector<double>>(
        numRows, std::vector<double>(numCols)));
    for (int i = 0; i < numSlices; ++i) {
        for (int j = 0; j < numRows; ++j) {
            for (int k = 0; k < numCols; ++k) {
                depthStream >> data.depth[i][j][k];
            }
        }
    }

    // Чтение velocity.txt
    velocityStream >> numSlices >> numRows >> numCols;
    data.velocity.resize(numSlices, std::vector<std::vector<double>>(
        numRows, std::vector<double>(numCols)));
    for (int i = 0; i < numSlices; ++i) {
        for (int j = 0; j < numRows; ++j) {
            for (int k = 0; k < numCols; ++k) {
                velocityStream >> data.velocity[i][j][k];
            }
        }
    }

    // Чтение relief.txt
    reliefStream >> numRows >> numCols;
    data.relief.resize(numRows, std::vector<double>(numCols));
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            reliefStream >> data.relief[i][j];
        }
    }

    return data;
}
// Функция для преобразования std::vector<std::vector<std::vector<double>>>
// в std::vector<Eigen::MatrixXd>
std::vector<Eigen::MatrixXd> convertToEigenMatrix(
    const std::vector<std::vector<std::vector<double>>>& input) {
    std::vector<Eigen::MatrixXd> matrices;
    matrices.reserve(input.size());
    for (const auto& slice : input) {
        int rows = static_cast<int>(slice.size());
        int cols = (rows > 0 ? static_cast<int>(slice[0].size()) : 0);
        Eigen::MatrixXd mat(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat(i, j) = slice[i][j];
            }
        }
        matrices.push_back(mat);
    }
    return matrices;
}
// Преобразование 2D-вектора в матрицу Eigen::MatrixXd
Eigen::MatrixXd convertToEigenMatrix2D(const std::vector<std::vector<double>>& mat) {
    int rows = static_cast<int>(mat.size());
    int cols = (rows > 0 ? static_cast<int>(mat[0].size()) : 0);
    Eigen::MatrixXd eigenMat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenMat(i, j) = mat[i][j];
        }
    }
    return eigenMat;
}


// Используемые typedef'ы для удобства
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Определяем alias для матрицы в формате RowMajor
template <typename T>
using rblock = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Функция сохранения матрицы формата rblock в файл в бинарном формате
void saveRBlockMatrix(const rblock<double>& mat, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Не удалось открыть файл для записи" << std::endl;
        return;
    }

    int32_t rows = static_cast<int32_t>(mat.rows());
    int32_t cols = static_cast<int32_t>(mat.cols());
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(mat.data()),
        rows * cols * sizeof(double));
}

// Функция загрузки матрицы формата rblock из бинарного файла
rblock<double> loadRBlockMatrix(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Ошибка открытия файла для чтения" << std::endl;
        return rblock<double>();  // Возвращаем пустую матрицу в случае ошибки
    }

    int32_t rows = 0;
    int32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    if (rows <= 0 || cols <= 0) {
        std::cerr << "Некорректные размеры матрицы" << std::endl;
        return rblock<double>();  // Возвращаем пустую матрицу в случае ошибки
    }

    rblock<double> mat(rows, cols);
    in.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));
    return mat;
}
