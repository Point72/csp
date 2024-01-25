#ifndef _IN_CSP_ENGINE_INPUTID_H
#define _IN_CSP_ENGINE_INPUTID_H

#include <bitset>
#include <limits>
#include <stdint.h>

namespace csp
{

using INOUT_ID_TYPE     = uint8_t;
using INOUT_ELEMID_TYPE = int32_t;

template<int TYPE>
struct InOutId
{
    InOutId( INOUT_ID_TYPE inputId, 
             INOUT_ELEMID_TYPE elementId = ELEM_ID_NONE ) : elemId( elementId ),
                                                            id( inputId )
    {}
    
    bool operator==( InOutId rhs ) const { return rhs.elemId == elemId && rhs.id == id; }

    static constexpr size_t maxId()             { return std::numeric_limits<INOUT_ID_TYPE>::max(); }
    static constexpr size_t maxInputs()         { return maxId() + 1; }
    static constexpr size_t maxOutputs()        { return maxId() + 1; }

    static constexpr size_t maxElemId()         { return std::numeric_limits<INOUT_ELEMID_TYPE>::max(); } //max is taken by NONE
    static constexpr size_t maxBasketElements() { return maxElemId() + 1; }

    static constexpr INOUT_ELEMID_TYPE ELEM_ID_NONE = -1;

    INOUT_ELEMID_TYPE elemId; //for basket inputs, if applicable
    INOUT_ID_TYPE     id;
};

using InputId  = InOutId<0>;
using OutputId = InOutId<1>;

};

#endif
