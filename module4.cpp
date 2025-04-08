
#include "ModelData.h"
#include <stdexcept>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXf;

namespace layer_2_grid{
    
/*!
 * \brief Вычисляет параметры прямой, проходящей через две точки.
 *
 * Функция вычисляет угловой коэффициент (наклон) и свободный член (сдвиг) прямой, которая проходит через
 * точки с координатами (\a x1, \a y1) и (\a x2, \a y2). Если разница между \a x1 и \a x2 меньше 1e-12,
 * функция рассматривает прямую как горизонтальную, устанавливая наклон равным 0 и свободный член равным максимуму
 * из \a y1 и \a y2.
 *
 * \tparam T Тип данных для вычислений.
 * \param[in] x1 Координата x первой точки.
 * \param[in] y1 Координата y первой точки.
 * \param[in] x2 Координата x второй точки.
 * \param[in] y2 Координата y второй точки.
 * \return Пара значений: первое значение — наклон прямой, второе значение — свободный член.
 */
template <typename T> std::pair<T, T> two_dots_line(T x1, T y1, T x2, T y2) {
  T a = T(0);
  T b = T(0);
  if (std::abs(x1 - x2) < 1e-12) {
    a = 0.0;
    b = std::max(y1, y2);
  } else {
    a = (y1 - y2) / static_cast<T>(x1 - x2);
    b = (x1 * y2 - y1 * x2) / (x1 - x2);
  }
  return {a, b};
}

/*!
 * \brief Выполняет линейную интерполяцию скорости по заданным точкам.
 * \param[in] z_size Длина вектора координат.
 * \param[in] dep Вектор глубин (без рельефа), должен быть отсортирован по возрастанию.
 * \param[in] vel Вектор скоростей. Размер должен быть на 1 больше, чем у dep.
 * \param[in] relief Значение рельефа (начальный уровень).
 * \param[in] v_const Опциональное постоянное значение скорости для точек, где z < relief.
 * \return Вектор интерполированных скоростей.
 */
template <typename T>
Eigen::VectorX<T>
interp_velocity(T z_size, const Eigen::Ref<const Eigen::VectorX<T>> &dep,
  const Eigen::Ref<const Eigen::VectorX<T>> &vel, T relief,
                std::optional<T> v_const = std::nullopt) {
  
  Eigen::VectorX<T> vel_interp = Eigen::VectorX<T>::Constant(
      z_size, v_const.has_value() ? v_const.value() : vel[0]);

  if (!std::is_sorted(dep.begin(), dep.end())) {
      throw std::invalid_argument("Вектор глубин должен быть отсортирован по возрастанию");
  }
  
  T a = T(0);
  T b = T(0);
  Eigen::Index j = relief;
  // Вычисляем коэффициенты для первого интервала
  std::tie(a, b) =
      two_dots_line(relief, vel(0), dep(0), vel(0 + 1));
  j++;
  while (j < dep[0]) {
      vel_interp(j) = a * j + b;
      ++j;
  }
  for (Eigen::Index i = 0; i < dep.size() - 1; ++i) {
    // Вычисляем коэффициенты для текущего интервала
    std::tie(a, b) =
        two_dots_line(dep(i), vel(i+1), dep(i + 1), vel(i + 2));

    // Пока значение j попадает в интервал [dep_full(i), dep_full(i+1)),
    // интерполируем
    while (j < z_size && j < dep(i + 1)) {
        vel_interp(j) =  a * j + b;
      ++j;
    }
    while (j < z_size) {
        vel_interp(j) = a * j + b;
        ++j;
    }
  }

  return vel_interp;
}

/*!
 * \brief Округляет значения рельефа до ближайшего кратного dz.
 * \param[in] relief Вектор значений рельефа для каждого столбца.
 * \param[in] dz Шаг округления.
 * \return static_relief.
 */
template <typename T>
Eigen::VectorX<T> get_static_relief(const Eigen::Ref<const Eigen::VectorX<T>> &relief, T dz) {
  return ((relief.array() / dz).round()) * dz;
}
/*!
 * \brief Формирует матрицу интерполированных скоростей для набора столбцов.
 * \param[in] z_min Минимальное значение вертикальной координаты.
 * \param[in] z_max Максимальное значение вертикальной координаты.
 * \param[in] dz Шаг сетки.
 * \param[in] depths Вектор матриц глубин для каждого слоя.
 * \param[in] velocities Вектор матриц скоростей для каждого слоя (размер = depths.size() + 1).
 * \param[in] relief Матрица значений рельефа.
 * \param[in] block_rows Количество строк в блоке.
 * \param[in] block_cols Количество столбцов в блоке.
 * \param[in] v_const Опциональное значение скорости для точек ниже рельефа.
 * \return Матрица, где каждая строка соответствует интерполированным скоростям для одного столбца.
 */
template <typename T>
Eigen::MatrixX<T> get_cube(T z_min, T z_max, T dz,
    const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& depths,
    const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& velocities,
    const Eigen::Ref<const Eigen::MatrixX<T>>& relief,
    int64_t block_rows, int64_t block_cols,
    const std::optional<T>& v_const) {
    // Вычисление числа элементов по оси z
    const T delta = z_max - z_min;
    const T epsilon = std::numeric_limits<T>::epsilon();
    const int64_t z_size = static_cast<int64_t>(std::ceil((delta - epsilon) / dz));

    // Количество пикселей
    const int64_t num_pixels = block_rows * block_cols;

    // Создание матрицы с размерами num_pixels x z_size
    Eigen::MatrixX<T> cube(num_pixels, z_size);

    const int64_t num_depth_slices = depths.size();

    // Цикл по пикселям
    for (int64_t r = 0; r < block_rows; ++r) {
        for (int64_t c = 0; c < block_cols; ++c) {
            int64_t idx = r * block_cols + c;

            // Формирование векторов глубин и скоростей
            Eigen::VectorX<T> dep_vec(num_depth_slices);
            Eigen::VectorX<T> vel_vec(num_depth_slices + 1);
            for (int64_t d = 0; d < num_depth_slices; ++d) {
                dep_vec(d) = depths[d](r, c);
            }
            for (int64_t d = 0; d < num_depth_slices + 1; ++d) {
                vel_vec(d) = velocities[d](r, c);
            }

            // Нормализация
            dep_vec = (dep_vec.array() - z_min) / dz;
            T rel_val = (relief(r, c) - z_min) / dz;

            // Заполнение строки матрицы развернутым вектором интерполированных значений
            cube.row(idx) = interp_velocity<T>(z_size, dep_vec, vel_vec, rel_val, v_const);
        }
    }

    return cube;
}

} // namespace layer_2_grid

template <class T>
using rblock =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*!
 * \brief Проверяет корректность входных данных для генератора.
 *
 * Функция выполняет серию проверок для обеспечения того, что все входные параметры
 * соответствуют ожидаемым требованиям. Если какой-либо параметр не проходит проверку,
 * выбрасывается исключение \c std::invalid_argument с описанием ошибки.
 *
 * \tparam T Тип элементов матриц (например, double, float).
 *
 * \param[in] blockRows Размер блока по строкам. Должен быть положительным целым числом.
 * \param[in] blockCols Размер блока по столбцам. Должен быть положительным целым числом.
 * \param[in] depths Вектор матриц глубин для каждого слоя. Не должен быть пустым, и все матрицы должны иметь одинаковые размеры.
 * \param[in] velocities Вектор матриц скоростей. Размер должен быть на единицу больше, чем у \a depths, и все матрицы должны иметь одинаковые размеры с \a depths.
 * \param[in] relief Матрица значений рельефа. Размеры должны совпадать с размерами матриц в \a depths.
 * \param[in] z_min Минимальное значение вертикальных координат. Должно быть меньше \a z_max.
 * \param[in] z_max Максимальное значение вертикальных координат. Должно быть больше \a z_min.
 * \param[in] dz Шаг сетки по вертикали. Должен быть положительным числом.
 * \param[in] v_const Опциональное постоянное значение скорости. Если задано, должно быть конечным числом.
 *
 * \throws std::invalid_argument Если какой-либо из параметров не соответствует требованиям.
 */
template <class T>
void validate_input_data(
    int64_t blockRows, int64_t blockCols,
    const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& depths,
    const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& velocities,
    const Eigen::Ref<const Eigen::MatrixX<T>>& relief,
    T z_min, T z_max, T dz,
    std::optional<T> v_const) {

    // Проверка blockRows и blockCols
    if (blockRows <= 0 || blockCols <= 0) {
        throw std::invalid_argument("blockRows и blockCols должны быть положительными целыми числами.");
    }

    // Проверка, что depths не пуст
    if (depths.empty()) {
        throw std::invalid_argument("Вектор depths не может быть пустым.");
    }

    // Проверка размеров матриц в depths
    auto rows = depths[0].rows();
    auto cols = depths[0].cols();
    for (const auto& d : depths) {
        if (d.rows() != rows || d.cols() != cols) {
            throw std::invalid_argument("Все матрицы в depths должны иметь одинаковые размеры.");
        }
    }

    // Проверка размеров матриц в velocities
    if (velocities.size() != depths.size() + 1) {
        throw std::invalid_argument("Размер velocities должен быть равен depths.size() + 1.");
    }
    for (const auto& v : velocities) {
        if (v.rows() != rows || v.cols() != cols) {
            throw std::invalid_argument("Все матрицы в velocities должны иметь одинаковые размеры с depths.");
        }
    }

    // Проверка размера relief
    if (relief.rows() != rows || relief.cols() != cols) {
        throw std::invalid_argument("Размер relief должен совпадать с размерами матриц в depths.");
    }

    // Проверка z_min, z_max, dz
    if (z_min >= z_max) {
        throw std::invalid_argument("z_min должен быть меньше z_max.");
    }
    if (dz <= 0) {
        throw std::invalid_argument("dz должен быть положительным числом.");
    }

    // Проверка v_const, если присутствует
    if (v_const.has_value() && !std::isfinite(v_const.value())) {
        throw std::invalid_argument("v_const должен быть конечным числом.");
    }
}
/*!
 * \brief Итератор по подматрицам для интерполяции.
 *
 * Класс реализует forward-итератор, который позволяет последовательно
 * обходить подматрицы (блоки) исходной матрицы. Для каждого блока выполняется
 * интерполяция данных с использованием заданных векторов глубин, скоростей, рельефа,
 * вертикальных координат и других параметров.
 *
 * \tparam T Тип элементов матрицы.
 */
template <class T> class submatrix_iterator {
public:
  // Типы, необходимые для forward iterator
  using value_type = rblock<T>;              /*!< Тип элемента*/
  using reference = rblock<T>; /*!< Тип ссылки на элемент*/
  using pointer = void; /*!< Тип указателя на элемент*/
  using iterator_category = std::forward_iterator_tag; /*!< Категория итератора*/

private:
    int64_t m_blockRows;          ///< Размер подматрицы по строкам.
    int64_t m_blockCols;          ///< Размер подматрицы по столбцам.
    int64_t m_numRowBlocks;       ///< Общее количество блоков по строкам.
    int64_t m_numColBlocks;       ///< Общее количество блоков по столбцам.
    int64_t m_currentRowBlock;    ///< Индекс текущего блока по строкам.
    int64_t m_currentColBlock;    ///< Индекс текущего блока по столбцам.

    int64_t m_start_row = 0;  ///< Индекс начала текущего блока по столбцам.
    int64_t m_start_col = 0; ///< Индекс начала текущего блока по столбцам.
    int64_t m_current_block_rows;  ///< Размер текущего блока по строкам.
    int64_t m_current_block_cols; ///< Размер текущего блока по столбцам.
    // Члены для интерполяции
    std::vector<Eigen::Ref<const Eigen::MatrixX<T>>> m_depths;      ///< Вектор матриц глубин для каждого столбца.
    std::vector<Eigen::Ref<const Eigen::MatrixX<T>>> m_velocities;  ///< Вектор матриц скоростей (размер = border_num + 1).
    Eigen::Ref<const Eigen::MatrixX<T>> m_relief;                   ///< Вектор значений рельефа для каждого столбца.
    T m_z_min;                                    ///< Минимальное значение вектора вертикальных координат.
    T m_z_max;                                    ///< Максимальное значение вектора вертикальных координат.
    T m_dz;                                       ///< Шаг сетки.
    std::optional<T> m_v_const;                   ///< Опциональное значение скорости для точек, где z < relief.

public:
 /*!
 * \param[in] blockRows Количество строк в блоке.
 * \param[in] blockCols Количество столбцов в блоке.
 * \param[in] depths Вектор матриц глубин для каждого слоя.
 * \param[in] velocities Вектор матриц скоростей для каждого слоя.
 * \param[in] relief Матрица рельефа.
 * \param[in] z_min Минимальная вертикальная координата.
 * \param[in] z_max Максимальная вертикальная координата.
 * \param[in] dz Шаг сетки.
 * \param[in] v_const Опциональное значение скорости для точек, где z < relief.
 * \param[in] currentRowBlock Начальный индекс блока по строкам (по умолчанию 0).
 * \param[in] currentColBlock Начальный индекс блока по столбцам (по умолчанию 0).
 */
  submatrix_iterator(int64_t blockRows, int64_t blockCols, 
                    const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& depths,
                    const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& velocities,
                    const Eigen::Ref<const Eigen::MatrixX<T>>& relief,
                    T z_min, T z_max, T dz, std::optional<T> v_const,
                    int64_t currentRowBlock = 0, int64_t currentColBlock = 0)
      :  m_blockRows(blockRows), m_blockCols(blockCols),
        m_depths(depths),
        m_velocities(velocities), m_relief(relief), m_z_min(z_min), m_z_max(z_max), m_dz(dz),
        m_v_const(v_const), m_currentRowBlock(currentRowBlock),
        m_currentColBlock(currentColBlock) {
        m_current_block_rows = relief.rows() < blockRows ? relief.rows() : blockRows;
        m_current_block_cols = relief.cols() < blockCols ? relief.cols() : blockCols;
    // Выполняем валидацию входных данных
    validate_input_data(blockRows, blockCols, depths, velocities, relief, z_min, z_max, dz, v_const);
    int64_t step_rows = blockRows;
    int64_t total_effective_rows = relief.rows();
    m_numRowBlocks = (total_effective_rows + step_rows - 1) / step_rows;

    int64_t step_cols = blockCols;
    int64_t total_effective_cols = relief.cols();
    m_numColBlocks = (total_effective_cols + step_cols - 1) / step_cols;
  }

  /*!
 * \return Куб (матрица) интерполированных скоростей для текущего блока.
 */
  reference operator*() {
      // Извлекаем блок из каждой матрицы глубин (m_depths)
      std::vector<Eigen::Ref<const Eigen::MatrixX<T>>> block_depths;
      for (const auto& d : m_depths) {
          block_depths.push_back(d.block(m_start_row, m_start_col, m_current_block_rows, m_current_block_cols));
      }

      // Извлекаем блок из каждой матрицы скоростей (m_velocities)
      std::vector<Eigen::Ref<const Eigen::MatrixX<T>>> block_velocities;
      for (const auto& v : m_velocities) {
          block_velocities.push_back(v.block(m_start_row, m_start_col, m_current_block_rows, m_current_block_cols));
      }

      // Извлекаем блок из матрицы рельефа
      Eigen::Ref<const Eigen::MatrixX<T>> block_relief = m_relief.block(m_start_row, m_start_col, m_current_block_rows, m_current_block_cols);

      // Выполняем интерполяцию для выбранного блока
      Eigen::MatrixX<T> cube = layer_2_grid::get_cube(
          m_z_min, m_z_max, m_dz, block_depths, block_velocities, block_relief,
          m_current_block_rows, m_current_block_cols, m_v_const);

      return cube;
  }

  // Оператор -> (не требуется, так как value_type — не объект)
  pointer operator->() const = delete; // Не поддерживается для MatrixXd

  /*!
   * \brief Префиксный оператор инкремента.
   *
   * Переходит к следующему блоку. При достижении конца строки переходит на следующую строку.
   *
   * \return Ссылка на обновленный итератор.
   */
  submatrix_iterator &operator++() {
    // Переходим к следующему столбцу
    m_currentColBlock++;
    // Если дошли до конца строки, переходим на следующую строку
    if (m_currentColBlock >= m_numColBlocks) {
      m_currentColBlock = 0;
      m_currentRowBlock++;
    }
     m_start_row = m_currentRowBlock * m_blockRows; //индекс строки
     m_start_col = m_currentColBlock * m_blockCols; // индекс столбца
     m_current_block_rows = (m_currentRowBlock == m_numRowBlocks - 1) //размер блока по строкам
        ? (m_relief.rows() - m_start_row)
        : m_blockRows;
     m_current_block_cols = (m_currentColBlock == m_numColBlocks - 1) // размер блока по столбцам
        ? (m_relief.cols() - m_start_col)
        : m_blockCols;
    return *this;
  }

  /*!
  * \brief Постфиксный оператор инкремента.
  *
  * Возвращает текущее состояние итератора, а затем переходит к следующему блоку.
  *
  * \return Копия итератора до инкремента.
  */
  submatrix_iterator operator++(int) {
    submatrix_iterator temp = *this; // Сохраняем текущее состояние
    ++(*this);   // Используем префиксный оператор
    return temp; // Возвращаем старое состояние
  }

  /*!
  * \brief Сравнение итераторов на равенство.
  *
  * \param other Другой итератор для сравнения.
  * \return true, если текущие индексы блоков совпадают, иначе false.
  */
  bool operator==(const submatrix_iterator &other) const {
    return m_currentRowBlock == other.m_currentRowBlock &&
           m_currentColBlock == other.m_currentColBlock;
  }

  /*!
  * \brief Сравнение итераторов на неравенство.
  *
  * \param other Другой итератор для сравнения.
  * \return true, если итераторы различны, иначе false.
  */
  bool operator!=(const submatrix_iterator &other) const {
    return !(*this == other);
  }
  /*!
   * \brief Возвращает индексы текущего блока.
   *
   * \return Пара целых чисел: индекс строки и индекс столбца текущего блока.
   */

  std::pair<int64_t, int64_t> index() { return {m_start_row, m_start_col}; }
  /*!
   * \brief Возвращает количество блоков по строкам и столбцам.
   *
   * \return Пара целых чисел: количество блоков по строкам и количество блоков по столбцам.
   */
  std::pair<int64_t, int64_t> shape() { return { m_current_block_rows , m_current_block_cols }; }
  /*!
  * \brief Создает итератор, указывающий на начало последовательности блоков.
  * \param[in] blockRows Количество строк в блоке.
  * \param[in] blockCols Количество столбцов в блоке.
  * \param[in] depths Вектор матриц глубин для каждого слоя.
  * \param[in] velocities Вектор матриц скоростей для каждого слоя.
  * \param[in] relief Матрица рельефа.
  * \param[in] z_min Минимальная вертикальная координата.
  * \param[in] z_max Максимальная вертикальная координата.
  * \param[in] dz Шаг сетки.
  * \param[in] v_const Опциональное значение скорости для точек, где z < relief.
  * \return Итератор на начало последовательности блоков.
  */
  static submatrix_iterator
  begin(int64_t blockRows, int64_t blockCols,
        const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& depths,
        const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& velocities,
        const Eigen::Ref<const Eigen::MatrixX<T>>&relief, T m_z_min, T m_z_max, T m_dz,
        std::optional<T> v_const) {
    return submatrix_iterator( blockRows, blockCols, depths, velocities, relief,
                              m_z_min, m_z_max, m_dz, v_const);
  }
  /*!
   * \brief Создает итератор, указывающий на конец последовательности блоков.
   *
   * \param[in] blockRows Количество строк в блоке.
   * \param[in] blockCols Количество столбцов в блоке.
   * \param[in] depths Вектор матриц глубин для каждого слоя.
   * \param[in] velocities Вектор матриц скоростей для каждого слоя.
   * \param[in] relief Матрица рельефа.
   * \param[in] z_min Минимальная вертикальная координата.
   * \param[in] z_max Максимальная вертикальная координата.
   * \param[in] dz Шаг сетки.
   * \param[in] v_const Опциональное значение скорости для точек, где z < relief.
   * \return Итератор на конец последовательности блоков.
   */
  static submatrix_iterator end(int64_t blockRows, int64_t blockCols,
                                const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& depths,
                                const std::vector<Eigen::Ref<const Eigen::MatrixX<T>>>& velocities,
                                const Eigen::Ref<const Eigen::MatrixX<T>>& relief,
                                  T z_min, T z_max, T dz, std::optional<T> v_const) {
    submatrix_iterator it(blockRows, blockCols,
                         depths, velocities, relief,  z_min, z_max, dz, v_const);
    it.m_currentRowBlock = it.m_numRowBlocks;
    it.m_currentColBlock = 0;
    return it;
  }
};
