#ifndef _IN_CSP_ENGINE_VECTORALLOCATOR_H
#define _IN_CSP_ENGINE_VECTORALLOCATOR_H

#include <csp/engine/CspType.h>

namespace csp
{
class VectorContainer
{
public:
    VectorContainer(const VectorContainer&) = delete;

    virtual ~VectorContainer(){};
    VectorContainer& operator=(const VectorContainer&) = delete;

    template< typename T >
    std::vector<T> &getVector()
    {
        return *reinterpret_cast<std::vector<T> *>(getVectorUntyped());
    }

    static std::unique_ptr<VectorContainer> createForCspType( CspTypePtr &type, bool optionalValues = true );
    static std::unique_ptr<VectorContainer> createForCspType( const CspType *type, bool optionalValues = true );
protected:
    VectorContainer() = default;
    virtual void *getVectorUntyped() = 0;
};

}

#endif
