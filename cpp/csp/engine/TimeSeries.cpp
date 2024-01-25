#include <csp/engine/TimeSeries.h>

namespace csp {
    using DuplicatePolicyEnum=TimeSeries::DuplicatePolicyEnum;
    INIT_CSP_ENUM(DuplicatePolicyEnum,
              "UNKNOWN",
              "LAST_VALUE",
              "FIRST_VALUE",
              "ALL_VALUES",
    );

}
