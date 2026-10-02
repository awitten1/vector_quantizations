#include "vq/pq.hpp"

namespace vq {

// Explicit instantiations so template errors surface when the library builds.
template class ProductQuantizer<float>;
template class ProductQuantizer<double>;

}  // namespace vq
