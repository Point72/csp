#include <csp/adapters/utils/MessageEnums.h>

namespace csp
{

INIT_CSP_ENUM( csp::adapters::utils::DateTimeWireType,
               "UNKNOWN",
               "UINT64_NANOS",
               "UINT64_MICROS",
               "UINT64_MILLIS",
               "UINT64_SECONDS"
);

INIT_CSP_ENUM( csp::adapters::utils::MsgProtocol,
               "UNKNOWN",
               "JSON",
               "PROTOBUF",
               "RAW_BYTES"
);

}
