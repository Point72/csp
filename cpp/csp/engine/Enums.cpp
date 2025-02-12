#include <csp/engine/Enums.h>

namespace csp
{

INIT_CSP_ENUM( PushMode, 
           "UNKNOWN",
           "LAST_VALUE",
           "NON_COLLAPSING",
           "BURST" 
    );


INIT_CSP_ENUM( ReplayMode, 
           "UNKNOWN",
           "EARLIEST",
           "LATEST",
           "START_TIME" 
    );

}
