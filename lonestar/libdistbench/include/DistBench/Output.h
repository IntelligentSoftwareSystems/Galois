#ifndef GALOIS_DISTBENCH_OUTPUT_H
#define GALOIS_DISTBENCH_OUTPUT_H

#include "galois/gIO.h"

#include <arrow/api.h>
#include <fstream>
#include <string>
#include <utility>

void writeOutput(const std::string& outputDir, arrow::Table& table);

template <typename T>
void writeOutput(const std::string& outputDir, const std::string& fieldName,
                 T* values, size_t length) {
  using ArrowType = typename arrow::CTypeTraits<T>::ArrowType;

  arrow::NumericBuilder<ArrowType> builder;

  auto appendStatus = builder.AppendValues(values, &values[length]);
  if (!appendStatus.ok()) {
    GALOIS_DIE("appending values", appendStatus.ToString());
  }

  std::shared_ptr<arrow::Array> array;

  auto finishStatus = builder.Finish(&array);
  if (!finishStatus.ok()) {
    GALOIS_DIE("finishing values", finishStatus.ToString());
  }

  auto typeFn = arrow::TypeTraits<ArrowType>::type_singleton();

  std::shared_ptr<arrow::Schema> schema =
      arrow::schema({arrow::field(fieldName, typeFn)});

  std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {array});

  writeOutput(outputDir, *table);
}

#endif
