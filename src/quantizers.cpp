#include "vq/pq.hpp"
#include "vq/sq.hpp"

namespace vq {

// Explicit instantiations so template errors surface when the library builds.
template class ProductQuantizer<float>;
template class ProductQuantizer<double>;
template class ScalarQuantizer<float>;
template class ScalarQuantizer<double>;

}  // namespace vq
