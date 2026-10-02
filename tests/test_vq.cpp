#include "check.hpp"
#include "vq/vq.hpp"

int main() {
  CHECK(!vq::version().empty());
  std::cout << "all tests passed\n";
  return EXIT_SUCCESS;
}
