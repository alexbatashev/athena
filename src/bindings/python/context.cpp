#include <athena/core/context/Context.h>

#include <pybind11/pybind11.h>

using namespace athena::core;

namespace polar::python {
class PYBIND11_EXPORT PyContext {
private:
  Context mContext; 
};
}

namespace py = pybind11;

PYBIND11_MODULE(polar, m) {
    py::class_<PyContext>(m, "Context")
        .def(py::init<const std::string &>());
}
