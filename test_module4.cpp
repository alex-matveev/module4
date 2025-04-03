#include "gtest/gtest.h"
#include "module4.cpp"  // Здесь объявлены layer2grid, SubmatrixIterator 
#include <chrono>
#include <optional>
#include <iostream>
#include <filesystem>
#include "ModelData.h" //  функции loadDepths, loadVelocities, loadRelief, loadRBlockMatrix...


TEST(Module4Test, CubeInterpolationMatchesExpectedResultFromFiles) {
    // Параметры интерполяции и разбиения
    const int border_num = 2;
    const std::optional<double> v_const = 100;
    const int blockRows = 220;
    const int blockCols = 600000;
    
    Eigen::VectorXd z = layer_2_grid::get_z_axis(-300.0, 250.0, 5.0);
  
    std::filesystem::path dataPath = "/home/matveevas/Downloads/Telegram Desktop/Module4";
    ModelData data =
        readModelData(dataPath / "depth.txt", dataPath / "velocity.txt",
                    dataPath / "relief.txt");
    auto flattened_depth = flatten3D(data.depth);
    auto flattened_velocity = flatten3D(data.velocity);
    const std::vector<VectorXd> &depths = convertToEigen(flattened_depth);
    const std::vector<VectorXd> &velocities = convertToEigen(flattened_velocity);
    VectorXd relief = flatten2D(data.relief);
    // После загрузки данных
    ASSERT_FALSE(depths.empty()) << "Failed to load depths";
    ASSERT_FALSE(velocities.empty()) << "Failed to load velocities";
 

    // Создаем результирующую матрицу (куб) с нулевыми значениями
    rblock<double> result = rblock<double>::Zero(z.size(), velocities[0].size());
    
    // Инициализируем итераторы для прохода по блокам
    auto begin = submatrix_iterator<double>::begin(
        result, blockRows, blockCols,
        depths, velocities, relief, border_num, z, v_const);
    auto end = submatrix_iterator<double>::end(
        result, blockRows, blockCols,
        depths, velocities, relief, border_num, z, v_const);
    
    // Проходим по всем блокам и заполняем результирующую матрицу
    for (auto it = begin; it != end; ++it) {
        auto block = *it;
        // Вычисляем координаты текущего блока в полной матрице
        int startRow = it.index().first * blockRows;
        int startCol = it.index().second * blockCols;
        result.block(startRow, startCol, block.rows(), block.cols()) = block;
    }
    // Считываем ожидаемый результат из бинарного файла
    rblock<double> expected = loadRBlockMatrix(dataPath / "result.bin");
    
    // Сравниваем размеры результирующей и ожидаемой матриц
    ASSERT_EQ(result.rows(), expected.rows()) << "Mismatch in number of rows.";
    ASSERT_EQ(result.cols(), expected.cols()) << "Mismatch in number of columns.";
    
    // Сравниваем значения элементов с допустимой погрешностью
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            EXPECT_NEAR(result(i, j), expected(i, j), 1e-6)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}
