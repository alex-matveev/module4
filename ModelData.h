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
struct ModelData {
  std::vector<std::vector<std::vector<double>>> depth;
  std::vector<std::vector<std::vector<double>>> velocity;
  std::vector<std::vector<double>> relief;
};

ModelData readModelData(const std::string &depthFile,
                        const std::string &velocityFile,
                        const std::string &reliefFile) {
  ModelData data;
  std::ifstream depthStream(depthFile);
  std::ifstream velocityStream(velocityFile);
  std::ifstream reliefStream(reliefFile);

  if (!depthStream.is_open() || !velocityStream.is_open() ||
      !reliefStream.is_open()) {
    std::cerr << "������ ��� �������� ������!" << std::endl;
    return data;
  }

  // ������ depth.txt
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

  // ������ velocity.txt
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

  // ������ relief.txt
  reliefStream >> numRows >> numCols;
  data.relief.resize(numRows, std::vector<double>(numCols));
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      reliefStream >> data.relief[i][j];
    }
  }

  return data;
}

// ������� ��� "�����������" 3D-������� � ������ ���������� ��������
std::vector<std::vector<double>>
flatten3D(const std::vector<std::vector<std::vector<double>>> &depth) {
  std::vector<std::vector<double>> flattened;
  flattened.reserve(depth.size());

  for (const auto &matrix : depth) {
    std::vector<double> flat;
    for (const auto &row : matrix) {
      flat.insert(flat.end(), row.begin(), row.end());
    }
    flattened.push_back(std::move(flat));
  }
  return flattened;
}
// ������� ��� "�����������" 2D-������� � ���������� ������ Eigen::VectorXd
Eigen::VectorXd flatten2D(const std::vector<std::vector<double>> &matrix) {
  // ������������ ����� ���������� ���������
  size_t total_elements = 0;
  for (const auto &row : matrix) {
    total_elements += row.size();
  }

  // ������� ���������� ������ ������� �������
  Eigen::VectorXd flat(total_elements);

  // ��������� flat ���������� �� matrix
  size_t index = 0;
  for (const auto &row : matrix) {
    for (const auto &element : row) {
      flat(index++) = element;
    }
  }

  return flat;
}
// ��� �������� ����� ������������ ��� typedef'�:
using Eigen::MatrixXd;
using Eigen::VectorXd;
void saveMatrix(const Eigen::MatrixXd &cube, const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "������ �������� ����� ��� ������" << std::endl;
    return;
  }
  // ��������� ������� ������� (���������� int32 ��� ������������� � Python)
  int32_t rows = static_cast<int32_t>(cube.rows());
  int32_t cols = static_cast<int32_t>(cube.cols());
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  // ����������� ������� � row-major �������������
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cube_rowmajor = cube;

  // ���������� ������ �������
  out.write(reinterpret_cast<const char *>(cube_rowmajor.data()),
            rows * cols * sizeof(double));
}
// ���������� alias ��� ������� � �������� �������� RowMajor:
template <class T>
using rblock =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void saveRBlockMatrix(const rblock<double> &mat, const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "������ �������� ����� ��� ������" << std::endl;
    return;
  }

  // ��������� ������� ������� (���������� int32_t ��� ������������� � Python)
  int32_t rows = static_cast<int32_t>(mat.rows());
  int32_t cols = static_cast<int32_t>(mat.cols());
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  // rblock<double> ��� ����� ������� �������� RowMajor, ������� ������ �����
  // ���������� ��������
  out.write(reinterpret_cast<const char *>(mat.data()),
            rows * cols * sizeof(double));
}
template <typename T>
using rblock = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

rblock<double> loadRBlockMatrix(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Ошибка открытия файла для чтения" << std::endl;
        return rblock<double>(); // Возвращаем пустую матрицу в случае ошибки
    }

    // Считываем размеры матрицы (предполагается, что они сохранены как int32_t)
    int32_t rows = 0;
    int32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Проверяем корректность считанных размеров
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Некорректные размеры матрицы" << std::endl;
        return rblock<double>(); // Возвращаем пустую матрицу в случае ошибки
    }

    // Создаем матрицу нужного размера
    rblock<double> mat(rows, cols);

    // Считываем данные матрицы
    in.read(reinterpret_cast<char*>(mat.data()), rows * cols * sizeof(double));

    return mat;
}

// ������� ��� ����������� std::vector<std::vector<double>> �
// std::vector<VectorXd>
std::vector<VectorXd>
convertToEigen(const std::vector<std::vector<double>> &input) {
  std::vector<VectorXd> depths;
  depths.reserve(input.size());
  for (const auto &vec : input) {
    // Eigen::Map ��������� ������� VectorXd, ��������� ������ �� vec
    depths.push_back(Eigen::Map<const VectorXd>(vec.data(), vec.size()));
  }
  return depths;
}
