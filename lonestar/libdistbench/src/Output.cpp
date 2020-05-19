#include "DistBench/Output.h"
#include "galois/runtime/Network.h"
#include "tsuba/tsuba_api.h"

#include <parquet/arrow/writer.h>
#include <iomanip>

namespace {

std::string zeroPad(int num, int width) {
  std::ostringstream out;

  out << std::setw(width) << std::setfill('0') << num;

  return out.str();
}

std::string makeOutputFilename(const std::string& outputDir) {
  std::string filename = zeroPad(galois::runtime::getHostID(), 8);

  std::string output{outputDir};
  if (output.empty() || output.compare(output.size() - 1, 1, "/") == 0) {
    output += filename;
  } else {
    output += "/" + filename;
  }

  return output;
}

} // namespace

void writeOutput(const std::string& dir, arrow::Table& table) {
  auto createResult = arrow::io::BufferOutputStream::Create();
  if (!createResult.ok()) {
    GALOIS_DIE("creating table", createResult.status().ToString());
  }

  std::shared_ptr<arrow::io::BufferOutputStream> out =
      createResult.ValueOrDie();

  auto writeStatus =
      parquet::arrow::WriteTable(table, arrow::default_memory_pool(), out, 4);
  if (!writeStatus.ok()) {
    GALOIS_DIE("writing table", writeStatus.ToString());
  }

  auto finishResult = out->Finish();
  if (!finishResult.ok()) {
    GALOIS_DIE("finishing buffer", finishResult.status().ToString());
  }

  std::shared_ptr<arrow::Buffer> buf = finishResult.ValueOrDie();

  std::string filename = makeOutputFilename(dir);

  int err = 0;
  if ((err = TsubaStore(filename.c_str(), buf->data(), buf->size())) != 0) {
    GALOIS_DIE("store", err);
  }
}
