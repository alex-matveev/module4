#include "gtest/gtest.h"
#include "module4.cpp"  // Здесь layer2grid, SubmatrixIterator 
#include <chrono>
#include <optional>
#include <iostream>
#include <filesystem>
#include "ModelData.h" //  функции convertToEigenMatrix, saveRBlockMatrix, loadRBlockMatrix

TEST(Module4Test, CubeInterpolationMatchesExpectedResultFromFiles) {
    // Параметры интерполяции и разбиения
    const std::optional<double> v_const = 100;
    const int blockRows = 30;
    const int blockCols = 20;
    
    double z_min = -300.0;
    double z_max = 250.0;
    double dz = 5.0;
    const double delta = z_max - z_min;
    const double epsilon = std::numeric_limits<double>::epsilon();

    const auto z_size = static_cast<int64_t>(std::ceil((delta - epsilon) / dz));
    std::string dataPath = "C:/Users/vicgu/source/repos/Module4/Module4/";
    ModelData data =
        readModelData(dataPath + "depth.txt", dataPath + "velocity.txt",
            dataPath + "relief.txt");
    /*
    std::filesystem::path dataPath = "C:/Users/vicgu/source/repos/Module4/Module4";
    ModelData data =
        readModelData(dataPath / "depth.txt", dataPath / "velocity.txt",
                    dataPath / "relief.txt");

    */

    std::vector<Eigen::MatrixXd> depths = convertToEigenMatrix(data.depth);
    std::vector<Eigen::MatrixXd> velocities = convertToEigenMatrix(data.velocity);
    Eigen::MatrixXd relief = convertToEigenMatrix2D(data.relief);

    // После загрузки данных
    ASSERT_FALSE(depths.empty()) << "Failed to load depths";
    ASSERT_FALSE(velocities.empty()) << "Failed to load velocities";
 

    // Создаем результирующую матрицу (куб) с нулевыми значениями
    rblock<double> result = rblock<double>::Constant(z_size, velocities[0].size(), 9999.0);

    // Создаем векторы ссылок на матрицы глубин
        std::vector<Eigen::Ref<const Eigen::MatrixXd>> ref_depths;
    for (auto& d : depths) {
        ref_depths.push_back(d);
    }

    // Создаем векторы ссылок на матрицы скоростей
    std::vector<Eigen::Ref<const Eigen::MatrixXd>> ref_velocities;
    for (auto& v : velocities) {
        ref_velocities.push_back(v);
    }

    // Создаем ссылку на матрицу рельефа
    Eigen::Ref<const Eigen::MatrixXd> ref_relief = relief;

    // Создаем итераторы с исправленными типами
    auto begin = submatrix_iterator<double>::begin(
        blockRows, blockCols,
        ref_depths, ref_velocities, ref_relief, z_min, z_max, dz, v_const);

    auto end = submatrix_iterator<double>::end(
        blockRows, blockCols,
        ref_depths, ref_velocities, ref_relief, z_min, z_max, dz, v_const);

    // Проходим по всем блокам и заполняем результирующую матрицу
    for (auto it = begin; it != end; ++it) {
        auto block = *it;

        // Вычисляем координаты текущего блока в полной матрице
        int startRow = it.index().first;
        int startCol = it.index().second;
        int total_cols = relief.cols();
        for (int i = 0; i < std::get<0>(it.shape()); i++) {
            for (int j = 0; j < std::get<1>(it.shape()); j++) {
                int global_row = startRow + i;
                int global_col = startCol + j;
                int global_index = global_row * total_cols + global_col;
                result.col(global_index) = block.col(i * std::get<1>(it.shape()) + j);
            }
        }
   
    }
    //saveRBlockMatrix(result, "result.bin");
    
    // Считываем ожидаемый результат из бинарного файла
    //rblock<double> expected = loadRBlockMatrix(dataPath / "result.bin");
    rblock<double> expected = loadRBlockMatrix(dataPath + "result.bin");

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
