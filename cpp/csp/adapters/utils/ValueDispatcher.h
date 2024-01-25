#ifndef _IN_CSP_ADAPTERS_PARQUET_ValueDispatcher_H
#define _IN_CSP_ADAPTERS_PARQUET_ValueDispatcher_H

#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

namespace csp
{
class PushBatch;
}

namespace csp::adapters::utils
{

using Symbol=std::variant<std::string,int64_t>;

template< typename V, typename ...SubscriberArgs >
class ValueDispatcher final
{
public:
    using ValueType = std::remove_reference_t<V>;
    using SubscriberType = std::function<void( ValueType *, SubscriberArgs... )>;
    using SubscriberContainer = std::vector<SubscriberType>;

    void addSubscriber( SubscriberType subscriber, std::optional<Symbol> symbol = {} )
    {
        if( symbol.has_value())
        {
            auto it = m_subscriberBySymbol.find( symbol.value());
            if( it == m_subscriberBySymbol.end())
            {
                it = m_subscriberBySymbol.emplace( symbol.value(), std::vector<SubscriberType>()).first;
            }
            it->second.push_back( subscriber );

        }
        else
        {
            m_subscribers.push_back( subscriber );
        }
    }

    SubscriberContainer *getSubscribers()
    {
        if( m_subscribers.empty())
        {
            return nullptr;
        }
        else
        {
            return &m_subscribers;
        }
    }

    SubscriberContainer *getSubscribersForSymbol( const Symbol &symbol )
    {
        auto it = m_subscriberBySymbol.find( symbol );
        if( it != m_subscriberBySymbol.end())
        {
            return &it->second;
        }
        else
        {
            return nullptr;
        }
    }

    void dispatch( ValueType *v, SubscriberContainer &subscriberContainer, SubscriberArgs... extraArgs )
    {
        for( auto &subscriber: subscriberContainer )
        {
            subscriber( v, extraArgs... );
        }
    }


    void dispatch( ValueType *v, SubscriberArgs... extraArgs )
    {
        dispatch( v, m_subscribers, extraArgs... );
    }

    void dispatch( ValueType *v, const Symbol *symbol, SubscriberArgs... extraArgs )
    {
        dispatch( v, m_subscribers, extraArgs... );
        if( symbol )
        {
            auto subscribers = getSubscribersForSymbol( *symbol );
            if( subscribers )
            {
                dispatch( v, *subscribers, extraArgs... );
            }
        }
    }

private:
    SubscriberContainer                             m_subscribers;
    std::unordered_map<Symbol, SubscriberContainer> m_subscriberBySymbol;
};

template< typename V >
using PushBatchValueDispatcher = ValueDispatcher<V, csp::PushBatch *>;

}
#endif
