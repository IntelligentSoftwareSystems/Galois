
namespace Galois {
  namespace Runtime {
    
class GaloisConfig {

  unsigned activeThreads;

 public:
  unsigned getActiveThreads() const {
    return activeThreads;
  }
  unsigned setActiveThreads() const {
    return activeThreads;
  }
};

GaloisConfig& getGaloisConfig();

  } // namespace Runtime
} // namespace Galois
