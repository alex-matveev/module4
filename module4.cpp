
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

/*! Функция interp_velocity :
 Выполняет линейную интерполяцию скорости по заданным точкам.
 \param[in] - z_size        : длина вектора координат
 \param[in] - dep       : вектор глубин (без рельефа);
 \param[in] - vel       : вектор скоростей. Его размер должен быть на 1 больше, чем у dep.
 \param[in] - relief    : значение рельефа (начальный уровень).
 \param[in] - v_const   : опциональное постоянное значение скорости для точек, где z < relief.
 \return - vel_interp : вектор интерполированных скоростей.
*/
template <typename T>
Eigen::VectorX<T>
interp_velocity(T z_size, const Eigen::Ref<const Eigen::VectorX<T>> &dep,
  const Eigen::Ref<const Eigen::VectorX<T>> &vel, T relief,
                std::optional<T> v_const = std::nullopt) {
  
  Eigen::VectorX<T> vel_interp = Eigen::VectorX<T>::Constant(
      z_size, v_const.has_value() ? v_const.value() : 0);

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

/*! Функция get_static_relief :
// Округляет значения рельефа до ближайшего кратного dz.
\return - static_relief     :
\param[in] - relief         : вектор значений рельефа для каждого столбца.

*/
template <typename T>
Eigen::VectorX<T> get_static_relief(const Eigen::Ref<const Eigen::VectorX<T>> &relief, T dz) {
  return ((relief.array() / dz).round()) * dz;
}
/*!
Функция get_cube:
 Формирует «куб» (матрицу) интерполированных скоростей для набора столбцов.
\return - cube     :матрица, где каждый столбец – это результат интерполяции
\param[in] - z_min         : минимальное значение вектора вертикальных координат.
\param[in] - z_max         : максимальное значение вектора вертикальных координат.
\param[in] - dz         : шаг сетки.
\param[in] - depths    : вектор из векторов глубин для каждого столбца.
\param[in] - velocities: вектор из векторов скоростей для каждого слоя.
\param[in] - relief    : вектор значений рельефа для каждого столбца.
\param[in] - size      : число столбцов (размер куба по второй оси).
\param[in] - v_const   : опциональное постоянное значение скорости для точек, где z < relief.

*/
template <typename T>
Eigen::MatrixX<T> get_cube(T z_min, T z_max, T dz,
    const std::vector<Eigen::MatrixX<T>>& depths,
    const std::vector<Eigen::MatrixX<T>>& velocities,
    const Eigen::MatrixX<T>& relief,
    int64_t block_rows, int64_t block_cols,
    const std::optional<T>& v_const) {
    const T delta = z_max - z_min;
    const T epsilon = std::numeric_limits<T>::epsilon();
    const int64_t z_size = static_cast<int64_t>(std::ceil((delta - epsilon) / dz));

    const int64_t num_pixels = block_rows * block_cols;
    Eigen::MatrixX<T> cube(z_size, num_pixels);

    // Число глубинных срезов определяется размером вектора depths
    const int64_t num_depth_slices = depths.size();

    // Проходим по всем пикселям блока (индексируем в порядке row-major)
    for (int64_t r = 0; r < block_rows; ++r) {
        for (int64_t c = 0; c < block_cols; ++c) {
            int64_t idx = r * block_cols + c;

            // Собираем векторы значений глубины и скорости для текущего пикселя из всех срезов
            Eigen::VectorX<T> dep_vec(num_depth_slices);
            Eigen::VectorX<T> vel_vec(num_depth_slices + 1);
            for (int64_t d = 0; d < num_depth_slices; ++d) {
                // Из каждого среза выбираем значение в координатах (r, c)
                dep_vec(d) = depths[d](r, c);
            }
            for (int64_t d = 0; d < num_depth_slices+1; ++d) {
                // Из каждого среза выбираем значение в координатах (r, c)
                vel_vec(d) = velocities[d](r, c);
            }
            // Нормализуем значения глубины относительно z_min и dz
            dep_vec = (dep_vec.array() - z_min) / dz;
            T rel_val = (relief(r, c) - z_min) / dz;

            // Интерполируем для данного пикселя.
            // Функция interp_velocity должна вернуть вектор длины z_size,
            // представляющий распределение интерполированных значений по оси z.
            cube.col(idx) = interp_velocity<T>(z_size, dep_vec, vel_vec, rel_val, v_const);
           // std::cout << "dep_vec" << dep_vec << std::endl;
            //std::cout << "vel_vec" << vel_vec << std::endl;
            //std::cout << "cube" << cube << std::endl;
        }
    }
    return cube;
}

} // namespace layer_2_grid

template <class T>
using rblock =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/*!
 * \brief Итератор по подматрицам для интерполяции.
 *
 * Данный класс реализует forward-итератор, который позволяет последовательно
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

    int64_t m_start_row = 0;
    int64_t m_start_col = 0;
    int64_t m_current_block_rows;
    int64_t m_current_block_cols;
    // Члены для интерполяции
    std::vector<Eigen::MatrixX<T>> m_depths;      ///< Вектор матриц глубин для каждого столбца.
    std::vector<Eigen::MatrixX<T>> m_velocities;  ///< Вектор матриц скоростей (размер = border_num + 1).
    Eigen::MatrixX<T> m_relief;                   ///< Вектор значений рельефа для каждого столбца.
    T m_z_min;                                    ///< Вектор вертикальных координат.
    T m_z_max;                                    ///< Вектор вертикальных координат.
    T m_dz;                                       ///< Вектор вертикальных координат.
    std::optional<T> m_v_const;                   ///< Опциональное значение скорости для точек, где z < relief.

public:
    /*!
      * \brief Конструктор итератора.
      *
      * Инициализирует итератор, вычисляя количество блоков по строкам и столбцам.
      *

      * \param blockRows Размер блока по строкам.
      * \param blockCols Размер блока по столбцам.
      * \param depths Вектор векторов глубин для каждого столбца.
      * \param velocities Вектор векторов скоростей (ожидается размер border_num + 1).
      * \param relief Вектор значений рельефа для каждого столбца.
      * \param z_min Вектор вертикальных координат.
      * \param z_max Вектор вертикальных координат.
      * \param dz Вектор вертикальных координат.
      * \param v_const Опциональное постоянное значение скорости для точек, где z < relief.
      * \param currentRowBlock Начальный индекс блока по строкам (по умолчанию 0).
      * \param currentColBlock Начальный индекс блока по столбцам (по умолчанию 0).
      */
  submatrix_iterator(int64_t blockRows, int64_t blockCols, 
                    const std::vector<Eigen::MatrixX<T>> &depths,
                    const std::vector<Eigen::MatrixX<T>> &velocities,
                    const Eigen::MatrixX<T> &relief,
                    T z_min, T z_max, T dz, std::optional<T> v_const,
                    int64_t currentRowBlock = 0, int64_t currentColBlock = 0)
      :  m_blockRows(blockRows), m_blockCols(blockCols),
        m_depths(depths),
        m_velocities(velocities), m_relief(relief), m_z_min(z_min), m_z_max(z_max), m_dz(dz),
        m_v_const(v_const), m_currentRowBlock(currentRowBlock),
        m_currentColBlock(currentColBlock) {
      m_current_block_rows = relief.rows() < blockRows ? relief.rows() : blockRows;
      m_current_block_cols = relief.cols() < blockCols ? relief.cols() : blockCols;
    int64_t step_rows = blockRows;
    int64_t total_effective_rows = relief.rows();
    m_numRowBlocks = (total_effective_rows + step_rows - 1) / step_rows;

    int64_t step_cols = blockCols;
    int64_t total_effective_cols = relief.cols();
    m_numColBlocks = (total_effective_cols + step_cols - 1) / step_cols;
  }

  /*!
   * \brief Возвращает текущую подматрицу для интерполяции.
   *
   * Оператор разыменования возвращает ссылку на подматрицу, сформированную из
   * сегментов исходной матрицы и заполненную интерполированными значениями
   * \return Ссылка на подматрицу (тип Eigen::Block<rblock<T>>).
   */
  reference operator*() {
      
      // Получаем блок исходной матрицы для заполнения
      //auto block = m_matrix.block(start_row, start_col, current_block_rows, current_block_cols);

      
      // Извлекаем блок из каждой матрицы глубин (m_depths)
      std::vector<Eigen::MatrixX<T>> block_depths;
      for (const auto& d : m_depths) {
          block_depths.push_back(d.block(m_start_row, m_start_col, m_current_block_rows, m_current_block_cols));
      }
      // Извлекаем блок из каждой матрицы скоростей (m_velocities)
      std::vector<Eigen::MatrixX<T>> block_velocities;
      for (const auto& v : m_velocities) {
          block_velocities.push_back(v.block(m_start_row, m_start_col, m_current_block_rows, m_current_block_cols));
      }
      // Извлекаем блок из матрицы рельефа
      Eigen::MatrixXd block_relief = m_relief.block(m_start_row, m_start_col, m_current_block_rows, m_current_block_cols);

      // Выполняем интерполяцию для выбранного блока
      Eigen::MatrixX<T> cube = layer_2_grid::get_cube(
          m_z_min, m_z_max, m_dz, block_depths, block_velocities, block_relief,
          m_current_block_rows, m_current_block_cols, m_v_const);
      // Присваиваем результат интерполяции блоку исходной матрицы
     // block = cube;
      
      

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
     m_start_row = m_currentRowBlock * m_blockRows;
     m_start_col = m_currentColBlock * m_blockCols;
     m_current_block_rows = (m_currentRowBlock == m_numRowBlocks - 1)
        ? (m_relief.rows() - m_start_row)
        : m_blockRows;
     m_current_block_cols = (m_currentColBlock == m_numColBlocks - 1)
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
     *
     * \param blockRows Размер блока по строкам.
     * \param blockCols Размер блока по столбцам.
     * \param depths Вектор векторов глубин для каждого столбца.
     * \param velocities Вектор векторов скоростей.
     * \param relief Вектор значений рельефа для каждого столбца.
     * \param z_min Минимальное значение вектора вертикальных координат.
     * \param z_max Максимальное значение вектора вертикальных координат.
     * \param dz Шаг сетки.
     * \param v_const Опциональное значение скорости.
     * \return Итератор, указывающий на первый блок.
     */
  static submatrix_iterator
  begin(int64_t blockRows, int64_t blockCols,
        const std::vector<Eigen::MatrixX<T>> &depths,
        const std::vector<Eigen::MatrixX<T>> &velocities,
        const Eigen::MatrixX<T> &relief, T m_z_min, T m_z_max, T m_dz,
        std::optional<T> v_const) {
    return submatrix_iterator( blockRows, blockCols, depths, velocities, relief,
                              m_z_min, m_z_max, m_dz, v_const);
  }
  /*!
   * \brief Создает итератор, указывающий на конец последовательности блоков.
   *
   * \param blockRows Размер блока по строкам.
   * \param blockCols Размер блока по столбцам.
   * \param depths Вектор векторов глубин для каждого столбца.
   * \param velocities Вектор векторов скоростей.
   * \param relief Вектор значений рельефа для каждого столбца.
   * \param z_min Минимальное значение вектора вертикальных координат.
   * \param z_max Максимальное значение вектора вертикальных координат.
   * \param dz Шаг сетки.
   * \param v_const Опциональное значение скорости.
   * \return Итератор, указывающий на конец последовательности блоков.
   */
  static submatrix_iterator end(int64_t blockRows, int64_t blockCols,
                               const std::vector<Eigen::MatrixX<T>> &depths,
                               const std::vector<Eigen::MatrixX<T>> &velocities,
                               const Eigen::MatrixX<T> &relief,
                               T z_min, T z_max, T dz, std::optional<T> v_const) {
    submatrix_iterator it(blockRows, blockCols,
                         depths, velocities, relief,  z_min, z_max, dz, v_const);
    it.m_currentRowBlock = it.m_numRowBlocks;
    it.m_currentColBlock = 0;
    return it;
  }
};
