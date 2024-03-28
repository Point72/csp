#include <csp/engine/VectorContainer.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/Struct.h>
#include <optional>

namespace
{
template< typename T >
class TypedVectorContainer : public csp::VectorContainer
{
protected:
    void *getVectorUntyped()
    {
        return &m_vector;
    }

private:
    std::vector<T> m_vector;
};
}

namespace csp
{

std::unique_ptr<VectorContainer> VectorContainer::createForCspType(CspTypePtr &type, bool optionalValues)
{
    return createForCspType(type.get());
}

std::unique_ptr<VectorContainer> VectorContainer::createForCspType( const CspType *type, bool optionalValues )
{
    return AllCspTypeSwitch::invoke(type, [optionalValues]( auto tag )
    {
        if(optionalValues)
        {
            return std::unique_ptr<VectorContainer>( new TypedVectorContainer<std::optional<typename decltype(tag)::type>> );
        }
        else
        {
            return std::unique_ptr<VectorContainer>( new TypedVectorContainer<typename decltype(tag)::type> );
        }
    });
}

}
