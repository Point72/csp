#ifndef _IN_CSP_CPPNODES_STATSIMPL_H
#define _IN_CSP_CPPNODES_STATSIMPL_H

#include <csp/engine/CppNode.h>
#include <csp/engine/WindowBuffer.h>

#include <functional>
#include <numeric>
#include <set>
#include <type_traits>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/ranked_index.hpp>

namespace csp::cppnodes
{

/*
General computation classes
*/

static constexpr double EPSILON = 1e-9;

class Count
{
    public:
        Count()
        {
            reset();
        }

        void add( double x )
        {
            m_count++;
        }

        void remove( double x )
        {
            m_count--;
        }

        void reset()
        {
            m_count = 0;
        }

        double compute() const
        {
            return static_cast<double>( m_count );
        }

    private:
        int64_t m_count;
};

class Sum
{
    public:
        Sum()
        {
            reset();
        }

        Sum( Sum && rhs ) = default;

        Sum & operator=( Sum && rhs ) = default;

        Sum & operator=( const Sum & rhs ) = delete;

        void add( double x )
        {
            m_sum += x;
        }

        void remove( double x )
        {
            m_sum -= x;
        }

        void reset()
        {
            m_sum = 0;
        }

        double compute() const
        {
            return m_sum;
        }

    private:
        double m_sum;
};

class KahanSum
{
    public:
        KahanSum()
        {
            reset();
        }

        void add( double x )
        {
            double y = x - m_comp;
            double tmp_sum = m_sum + y;
            m_comp = tmp_sum - m_sum - y;
            m_sum = tmp_sum;
        }

        void remove( double x )
        {
            // same as adding negative
            double y = -x - m_comp;
            double tmp_sum = m_sum + y;
            m_comp = tmp_sum - m_sum - y;
            m_sum = tmp_sum;
        }

        void reset()
        {
            m_sum = m_comp = 0;
        }

        double compute() const
        {
            return m_sum;
        }

    private:
        double m_sum;
        double m_comp;

};

class Mean
{
    public:
        Mean()
        {
            reset();
        }

        void add( double x )
        {
            m_count++;
            m_mean += ( x - m_mean ) / m_count;
        }

        void remove( double x )
        {
            m_count--;
            if( m_count > 0 )
                m_mean += ( m_mean - x ) / m_count;
            else
                m_mean = 0;
        }

        void reset()
        {
            m_mean = m_count = 0;
        }

        double compute() const
        {
            if( m_count > 0 )
                return m_mean;
            return std::numeric_limits<double>::quiet_NaN();
        }

    private:
        double m_mean;
        int64_t m_count;
};

class First
{
    public:
        void add( double x )
        {
            m_value_buffer.push( x );
        }

        void remove( double x )
        {
            m_value_buffer.pop_left();
        }

        void reset()
        {
            m_value_buffer.clear();
        }

        double compute()
        {
            if( !m_value_buffer.empty() )
            {
                return m_value_buffer[-1];
            }
            return std::numeric_limits<double>::quiet_NaN();
        }

    private:
        VariableSizeWindowBuffer<double> m_value_buffer;
};

class Last
{
    public:
        Last()
        {
            reset();
        }

        void add( double x )
        {
            m_last = x;
            m_count++;
        }

        void remove( double x )
        {
            m_count--;
        }

        void reset()
        {
            m_count = 0;
        }

        double compute()
        {
            if( m_count > 0 )
                return m_last;
            return std::numeric_limits<double>::quiet_NaN();
        }

    private:
        double m_last;
        double m_count;
};

class Unique
{
    public:

        Unique()
        {
            m_powPrecision = 1;
        }

        Unique( int64_t precision )
        {
            m_powPrecision = pow( 10.0, ( double ) precision );
        }

        void add( double x )
        {
            int64_t rounded_result = ( int64_t )( x * m_powPrecision );
            m_dict[rounded_result]++;
        }

        void remove( double x )
        {
            int64_t rounded_result = ( int64_t )( x * m_powPrecision );
            auto it = m_dict.find( rounded_result );
            if( it -> second == 1 )
                m_dict.erase( rounded_result );
            else
                it -> second--;
        }

        void reset()
        {
            m_dict.clear();
        }

        double compute() const
        {
            return static_cast<double>( m_dict.size() );
        }

    private:

        std::unordered_map<int64_t, int64_t> m_dict;
        double m_powPrecision;
};

class Product
{
    public:
        Product()
        {
            reset();
        }

        void add( double x )
        {
            m_count++;
            if( x == 0 )
                m_nzero++;
            else
                m_prod *= x;
        }

        void remove( double x )
        {
            m_count--;
            if( x == 0 )
                m_nzero--;
            else
                m_prod /= x;
        }

        void reset()
        {
            m_prod = 1.0;
            m_count = m_nzero = 0;
        }

        double compute() const
        {
            if( m_count > 0 )
            {
                if( m_nzero == 0 )
                    return m_prod;
                else
                    return 0;
            }
            else
                return std::numeric_limits<double>::quiet_NaN();
        }

    private:

        double m_prod;
        int64_t m_count;
        int64_t m_nzero;
};

class WeightedMean
{
    public:
        WeightedMean()
        {
            reset();
        }

        void add( double x, double w )
        {
            m_weight += w;
            if( m_weight > EPSILON )
                m_wmean += ( x * w - w * m_wmean ) / m_weight;
        }

        void remove( double x, double w )
        {
            m_weight -= w;
            if( m_weight > EPSILON )
                m_wmean -= ( x * w - w * m_wmean ) / m_weight;
            else
                m_wmean = m_weight = 0;
        }

        void reset()
        {
            m_wmean = m_weight = 0;
        }

        double compute() const
        {
            if( m_weight > EPSILON )
            {
                return m_wmean;
            }
            return std::numeric_limits<double>::quiet_NaN();
        }

    private:
        double m_wmean;
        double m_weight;
};

class Variance
{
    public:
        Variance()
        {
            m_ddof = 1;
            reset();
        }

        Variance( int64_t ddof ) : Variance()
        {
            m_ddof = ddof;
        }

        void add( double x )
        {
            // Track consecutive values to avoid numerical errors when all values are identical
            // This approach is taken from the pandas rolling variance logic 
            m_consecutiveValueCount = ( m_consecutiveValueCount && x == m_lastValue ? m_consecutiveValueCount + 1 : 1 );
            m_lastValue = x;

            m_count++;
            m_dx = x - m_mean;
            m_mean += m_dx / m_count;
            m_unnormVar += ( x - m_mean ) * m_dx;
        }

        void remove( double x )
        {
            m_count--;
            if( m_count == 0 )
            {
                m_mean = m_unnormVar = 0;
                return;
            }
            m_dx = x - m_mean;
            m_mean -= m_dx / m_count;
            m_unnormVar -= ( x - m_mean ) * m_dx;
        }

        void reset()
        {
            m_mean = m_unnormVar = m_count = 0;
            m_consecutiveValueCount = 0;
        }

        double compute() const
        {
            if( m_count > m_ddof )
            {
                // Special case for homogeneous window, modelled off of pandas impl
                if( m_consecutiveValueCount >= m_count ) [[unlikely]]
                    return 0;
                return ( m_unnormVar < 0 ? 0 : m_unnormVar / ( m_count - m_ddof ) );
            }

            return std::numeric_limits<double>::quiet_NaN();
        }

    private:

        double m_mean;
        double m_unnormVar;
        double m_dx;
        double m_count;
        int64_t m_ddof;

        // Below variables are used to eliminate numerical errors when all values in the window are identical
        double m_lastValue;
        int64_t m_consecutiveValueCount;
};

class WeightedVariance
{
    public:
        WeightedVariance()
        {
            m_ddof = 1;
            reset();
        }

        WeightedVariance( int64_t ddof ) : WeightedVariance()
        {
            m_ddof = ddof;
        }

        void add( double x, double w )
        {
            if( w <= 0 )
                return;
            
            // See comment in Variance::add on handling homogeneous data streams
            m_consecutiveValueCount = ( m_consecutiveValueCount && x == m_lastValue ? m_consecutiveValueCount + 1 : 1 );
            m_lastValue = x;

            m_count++;
            m_wsum += w;
            m_dx = x - m_wmean;
            m_wmean += ( w / m_wsum ) * m_dx;
            m_unnormWVar += w * ( x - m_wmean ) * m_dx;
        }

        void remove( double x, double w )
        {
            if( w <= 0 )
                return;
            
            m_count--;
            m_wsum -= w;
            if( m_wsum < EPSILON )
            {
                m_wsum = m_wmean = m_unnormWVar = 0;
                return;
            }
            m_dx = x - m_wmean;
            m_wmean -= ( w / m_wsum ) * m_dx;
            m_unnormWVar -= w * ( x - m_wmean ) * m_dx;
        }

        void reset()
        {
            m_wsum = m_wmean = m_unnormWVar = 0;
            m_consecutiveValueCount = m_count = 0;
        }

        double compute() const
        {
            if( m_wsum > m_ddof )
            {
                // Special case for homogeneous window, modelled off of pandas impl
                if( m_consecutiveValueCount >= m_count ) [[unlikely]]
                    return 0;
                return ( m_unnormWVar < 0 ? 0 : m_unnormWVar / ( m_wsum - m_ddof ) );
            }

            return std::numeric_limits<double>::quiet_NaN();
        }

    private:

        double m_wsum;
        double m_wmean;
        double m_unnormWVar;
        double m_dx;
        int64_t m_ddof;

        // Below variables are used to eliminate numerical errors when all values in the window are identical
        int64_t m_count;
        double m_lastValue;
        int64_t m_consecutiveValueCount;
};

class Covariance
{
    public:
        Covariance()
        {
            m_ddof = 1;
            reset();
        }

        Covariance( int64_t ddof ) : Covariance()
        {
            m_ddof = ddof;
        }

        void add( double x, double y )
        {
            m_count++;
            m_dx = x - m_mux;
            m_mux += m_dx / m_count;
            m_muy += ( y - m_muy ) / m_count;
            m_unnormCov += m_dx * ( y - m_muy );
        }

        void remove( double x, double y )
        {
            m_count--;
            if( m_count == 0 )
            {
                m_mux = m_muy = m_unnormCov = 0;
                return;
            }

            m_dx = x - m_mux;
            m_mux -= m_dx / m_count;
            m_muy -= ( y - m_muy ) / m_count;
            m_unnormCov -= m_dx * ( y - m_muy );
        }

        void reset()
        {
            m_mux = m_muy = m_unnormCov = m_count = 0;
        }

        double compute() const
        {
            return ( m_count > m_ddof ? m_unnormCov / ( m_count - m_ddof ) : std::numeric_limits<double>::quiet_NaN() );
        }

    private:

        double m_mux;
        double m_muy;
        double m_unnormCov;
        double m_dx;
        double m_count;
        int64_t m_ddof;
};

class WeightedCovariance
{
    public:
        WeightedCovariance()
        {
            m_ddof = 1;
            reset();
        }

        WeightedCovariance( int64_t ddof ) : WeightedCovariance()
        {
            m_ddof = ddof;
        }

        void add( double x, double y, double w )
        {
            if( w <= 0 )
                return;
            m_wsum += w;
            m_dx = x - m_wmux;
            m_wmux += ( w / m_wsum ) * m_dx;
            m_wmuy += ( w / m_wsum ) * ( y - m_wmuy );
            m_unnormWCov += w * m_dx * ( y - m_wmuy );
        }

        void remove( double x, double y, double w )
        {
            m_wsum -= w;
            if( m_wsum < EPSILON )
            {
                m_wsum = m_wmux = m_wmuy = m_unnormWCov = 0;
                return;
            }
            m_dx = x - m_wmux;
            m_wmux -= ( w / m_wsum ) * m_dx;
            m_wmuy -= ( w / m_wsum ) * ( y - m_wmuy );
            m_unnormWCov -= w * m_dx * ( y - m_wmuy );
        }

        void reset()
        {
            m_wmux = m_wmuy = m_unnormWCov = m_wsum = 0;
        }

        double compute() const
        {
            return ( m_wsum > m_ddof ? m_unnormWCov / ( m_wsum - m_ddof ) : std::numeric_limits<double>::quiet_NaN() );
        }

    private:

        double m_wmux;
        double m_wmuy;
        double m_unnormWCov;
        double m_dx;

        double m_wsum;
        int64_t m_ddof;
};

double corrCompute( double cov, double vx, double vy )
{
    if( vx < EPSILON || vy < EPSILON )
        return std::numeric_limits<double>::quiet_NaN();
    return cov / sqrt( vx * vy );
}

class Correlation
{
    public:
        Correlation()
        {
            reset();
        }

        Correlation( int64_t arg ) : Correlation() { }

        void add( double x, double y )
        {
            m_cov.add( x, y );
            m_vx.add( x );
            m_vy.add( y );
        }

        void remove( double x, double y )
        {
            m_cov.remove( x, y );
            m_vx.remove( x );
            m_vy.remove( y );
        }

        void reset()
        {
            m_cov.reset();
            m_vx.reset();
            m_vy.reset();
        }

        double compute() const
        {
            return corrCompute( m_cov.compute(), m_vx.compute(), m_vy.compute() );
        }

    private:
        Covariance m_cov;
        Variance   m_vx;
        Variance   m_vy;
};

class WeightedCorrelation
{
    public:
        WeightedCorrelation()
        {
            reset();
        }

        WeightedCorrelation( int64_t arg ) : WeightedCorrelation() { }

        void add( double x, double y, double w )
        {
            m_cov.add( x, y, w );
            m_vx.add( x, w );
            m_vy.add( y, w );
        }

        void remove( double x, double y, double w )
        {
            m_cov.remove( x, y, w );
            m_vx.remove( x, w );
            m_vy.remove( y, w );
        }

        void reset()
        {
            m_cov.reset();
            m_vx.reset();
            m_vy.reset();
        }

        double compute() const
        {
            return corrCompute( m_cov.compute(), m_vx.compute(), m_vy.compute() );
        }

    private:
        WeightedCovariance m_cov;
        WeightedVariance   m_vx;
        WeightedVariance   m_vy;
};

class StandardError
{
    public:
        StandardError()
        {
            m_ddof = 1;
            reset();
        }

        StandardError( int64_t ddof ) : StandardError()
        {
            m_ddof = ddof;
            m_var = Variance( ddof );
        }

        void add( double x )
        {
            m_count++;
            m_var.add( x );
        }

        void remove( double x )
        {
            m_count--;
            m_var.remove( x );
        }

        void reset()
        {
            m_count = 0;
            m_var.reset();
        }

        double compute() const
        {
            return ( m_count > m_ddof ? sqrt( m_var.compute() / ( m_count - m_ddof ) ) : std::numeric_limits<double>::quiet_NaN() );
        }

    private:
        Variance m_var;
        int64_t m_ddof;
        double m_count;
};

class WeightedStandardError
{
    public:
        WeightedStandardError()
        {
            m_ddof = 1;
            reset();
        }

        WeightedStandardError( int64_t ddof ) : WeightedStandardError()
        {
            m_ddof = ddof;
            m_var = WeightedVariance( ddof );
        }

        void add( double x, double w )
        {
            m_wsum += w;
            m_var.add( x, w );
        }

        void remove( double x, double w )
        {
            m_wsum -= w;
            if( m_wsum < EPSILON )
                m_wsum = 0;
            m_var.remove( x, w );
        }

        void reset()
        {
            m_wsum = 0;
            m_var.reset();
        }

        double compute() const
        {
            return ( m_wsum > m_ddof && m_wsum > EPSILON ? sqrt( m_var.compute() / ( m_wsum - m_ddof ) ) : std::numeric_limits<double>::quiet_NaN() );
        }

    private:
        WeightedVariance m_var;
        int64_t m_ddof;
        double m_wsum;
};



double skewCompute( double count, double mx, double mx3, double vx, bool bias )
{
    if( count <= 2 || vx < EPSILON )
            return std::numeric_limits<double>::quiet_NaN();

    double bias_skew = ( mx3 - 3 * mx * vx - mx * mx * mx ) / ( vx * sqrt( vx ) );
    if( bias )
        return bias_skew;

    double factor = sqrt( count * ( count - 1 ) ) / ( count- 2 );
    return factor * bias_skew;
}

double kurtCompute( double count, double mx, double mx2, double mx3, double mx4, double vx, bool bias, bool excess )
{
    if( count <= 3 || vx < EPSILON )
        return std::numeric_limits<double>::quiet_NaN();

    double bias_kurt = ( mx4 - 4 * mx * mx3 + 6 * mx2 * mx * mx - 3 * mx * mx * mx * mx ) / ( vx * vx );

    if( bias )
    {
        if( excess )
            return bias_kurt-3;
        return bias_kurt;
    }

    double kfactor = ( count + 1 ) * ( count - 1 ) / ( ( count - 2 ) * ( count - 3 ) );
    double ub_kurt = kfactor * bias_kurt;
    double gfactor = ( count - 1 ) / ( count + 1 ) * kfactor;
    ub_kurt -= 3 * gfactor;
    if( !excess )
        ub_kurt += 3;
        
    return ub_kurt;
}

class Skew
{
    public:
        Skew()
        {
            m_vx = Variance( 0 );
            m_bias = false;
            reset();
        }

        Skew( bool bias ) : Skew()
        {
            m_bias = bias;
        }

        void add( double x )
        {
            m_count++;
            m_mx.add( x );
            m_mx3.add( x*x*x );
            m_vx.add( x );
        }

        void remove( double x )
        {
            m_count--;
            m_mx.remove( x );
            m_mx3.remove( x*x*x );
            m_vx.remove( x );
        }

        void reset()
        {
            m_mx.reset();
            m_mx3.reset();
            m_vx.reset();
            m_count = 0;
        }

        double compute() const
        {
            return skewCompute( m_count, m_mx.compute(), m_mx3.compute(), m_vx.compute(), m_bias );
        }

    private:

        Mean m_mx;
        Mean m_mx3;
        Variance m_vx;
        double m_count;
        bool m_bias;
};

class WeightedSkew
{
    public:
        WeightedSkew()
        {
            m_vx = WeightedVariance( 0 );
            m_bias = false;
            reset();
        }

        WeightedSkew( bool bias ) : WeightedSkew()
        {
            m_bias = bias;
        }

        void add( double x, double w )
        {
            m_count++;
            m_mx.add( x, w );
            m_mx3.add( x*x*x, w );
            m_vx.add( x, w );
        }

        void remove( double x, double w )
        {
            m_count--;
            m_mx.remove( x, w );
            m_mx3.remove( x*x*x, w );
            m_vx.remove( x, w );
        }

        void reset()
        {
            m_mx.reset();
            m_mx3.reset();
            m_vx.reset();
            m_count = 0;
        }

        double compute() const
        {
            return skewCompute( m_count, m_mx.compute(), m_mx3.compute(), m_vx.compute(), m_bias );
        }

    private:

        WeightedMean m_mx;
        WeightedMean m_mx3;
        WeightedVariance m_vx;
        double m_count;
        bool m_bias;
};

class Kurtosis
{
    public:
        Kurtosis()
        {
            m_vx = Variance( 0 );
            m_bias = false;
            m_excess = true;
            reset();
        }

        Kurtosis( bool bias, bool excess ) : Kurtosis()
        {
            m_bias = bias;
            m_excess = excess;
        }

        void add( double x )
        {
            m_count++;
            double val = x;
            m_mx.add( val );
            m_vx.add( val );
            val *= x;
            m_mx2.add( val );
            val *= x;
            m_mx3.add( val );
            val *= x;
            m_mx4.add( val );

        }

        void remove( double x )
        {
            m_count--;
            double val = x;
            m_mx.remove( val );
            m_vx.remove( val );
            val *= x;
            m_mx2.remove( val );
            val *= x;
            m_mx3.remove( val );
            val *= x;
            m_mx4.remove( val );
        }

        void reset()
        {
            m_mx.reset();
            m_mx2.reset();
            m_mx3.reset();
            m_mx4.reset();
            m_vx.reset();
            m_count = 0;
        }

        double compute() const
        {
            return kurtCompute( m_count, m_mx.compute(), m_mx2.compute(), m_mx3.compute(), m_mx4.compute(), m_vx.compute(), m_bias, m_excess );
        }

    private:

        Mean m_mx;
        Mean m_mx2;
        Mean m_mx3;
        Mean m_mx4;
        Variance m_vx;
        double m_count;
        bool m_bias;
        bool m_excess;
};

class WeightedKurtosis
{
    public:
        WeightedKurtosis()
        {
            m_vx = WeightedVariance( 0 );
            m_bias = false;
            m_excess = true;
            reset();
        }

        WeightedKurtosis( bool bias, bool excess ) : WeightedKurtosis()
        {
            m_bias = bias;
            m_excess = excess;
        }

        void add( double x, double w )
        {
            m_count++;
            double val = x;
            m_mx.add( val, w );
            m_vx.add( val, w );
            val *= x;
            m_mx2.add( val, w );
            val *= x;
            m_mx3.add( val, w );
            val *= x;
            m_mx4.add( val, w );
        }

        void remove( double x, double w )
        {
            m_count--;
            double val = x;
            m_mx.remove( val, w );
            m_vx.remove( val, w );
            val *= x;
            m_mx2.remove( val, w );
            val *= x;
            m_mx3.remove( val, w );
            val *= x;
            m_mx4.remove( val, w );
        }

        void reset()
        {
            m_mx.reset();
            m_mx2.reset();
            m_mx3.reset();
            m_mx4.reset();
            m_vx.reset();
            m_count = 0;
        }

        double compute() const
        {
            return kurtCompute( m_count, m_mx.compute(), m_mx2.compute(), m_mx3.compute(), m_mx4.compute(), m_vx.compute(), m_bias, m_excess );
        }

    private:

        WeightedMean m_mx;
        WeightedMean m_mx2;
        WeightedMean m_mx3;
        WeightedMean m_mx4;
        WeightedVariance m_vx;
        double m_count;
        bool m_bias;
        bool m_excess;
};

template <typename Comparator>
using ost = boost::multi_index::multi_index_container<double, boost::multi_index::indexed_by<boost::multi_index::ranked_non_unique<boost::multi_index::identity<double>, Comparator>>>;

class Quantile
{
    enum Interpolate
    {
        LINEAR, LOWER, HIGHER, MIDPOINT, NEAREST
    };

    public:
        Quantile( const std::vector<Dictionary::Data> & quants, int64_t interpolation )
        {
            m_quants = quants;
            m_interpolation = interpolation;
        }

        Quantile( Quantile && rhs )
        {
            m_quants = rhs.m_quants;
            m_interpolation = rhs.m_interpolation;
            m_tree = std::move( rhs.m_tree );
        }

        Quantile & operator=( Quantile && rhs )
        {
            m_quants = rhs.m_quants;
            m_interpolation = rhs.m_interpolation;
            m_tree = std::move( rhs.m_tree );

            return *this;
        }

        Quantile() = default;
        Quantile( const Quantile & rhs ) = delete;
        Quantile & operator=( const Quantile & rhs ) = delete;

        void add( double x )
        {
            m_tree.insert( x );
        }

        void remove( double x )
        {
            m_tree.erase( m_tree.find( x ) );
        }

        void reset()
        {
            m_tree.clear();
        }

        double compute( int index ) const
        {
            // Compute which values to find
            if( m_tree.size() == 0 )
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            double target = std::get<double>( m_quants[index]._data ) * ( m_tree.size() - 1 );
            int ft = floor( target );
            int ct = ceil( target );
            auto fIt = m_tree.get<0>().nth( ft );
            auto cIt = ( ft == ct ) ? fIt : std::next( fIt );

            double qtl = 0.0;
            switch ( m_interpolation )
            {
            case LINEAR:
                if ( ft == target )
                {
                    qtl = *fIt;
                }
                else
                {
                    double lower = *fIt;
                    double higher = *cIt;
                    qtl = ( 1 - target + ft ) * lower + ( 1 - ct + target ) * higher;
                }
                break;
            case LOWER:
                qtl = *fIt;
                break;
            case HIGHER:
                qtl = *cIt;
                break;
            case MIDPOINT:
                if ( ft == target )
                {
                    qtl = *fIt;
                }
                else
                {
                    double lower = *fIt;
                    double higher = *cIt;
                    qtl = ( higher + lower ) / 2;
                }
                break;
            case NEAREST:
                if ( target - ft < ct - target )
                {
                    qtl = *fIt;
                }
                else
                {
                    qtl = *cIt;
                }
                break;
            default:
                break;
            }
            return qtl;
        }

    private:
        ost<std::less<double>> m_tree;
        std::vector<Dictionary::Data> m_quants;
        int64_t m_interpolation;
};

class AscendingMinima
{
public:
    AscendingMinima( bool max ) : m_max( max ) {}
    AscendingMinima( AscendingMinima && rhs ) = default;
    AscendingMinima & operator=( AscendingMinima && rhs ) = default;

    AscendingMinima() = default;
    AscendingMinima( const AscendingMinima & rhs ) = delete;

    void add( double x )
    {
        while( !m_value_buffer.empty() && ( m_max ? x > m_value_buffer[0] : x < m_value_buffer[0] ) )
        {
            m_value_buffer.pop_right();
        }
        m_value_buffer.push( x );
    }

    void remove( double x )
    {
        if( x == m_value_buffer[-1] )
        {
            m_value_buffer.pop_left();
        }
    }

    void reset()
    {
        m_value_buffer.clear();
    }

    double compute()
    {
        if( m_value_buffer.empty() )
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return m_value_buffer[-1];
    }

private:
    bool m_max;
    VariableSizeWindowBuffer<double> m_value_buffer;
};

class Rank
{
    enum RankMethod
    {
        MIN, MAX, AVG
    };

    enum NanOption
    {
        KEEP, LAST
    };

    public:
        Rank() = default;

        Rank( int64_t method, int64_t nanopt )
        {
            m_method = method;
            m_nanopt = nanopt;
        }

        // no copy as always
        Rank( Rank && rhs ) = default;
        Rank & operator=( Rank && rhs ) = default;
        Rank( const Rank & rhs ) = delete;
        Rank & operator=( const Rank & rhs ) = delete;

        void add( double x )
        {
            if( unlikely( isnan( x ) ) )
            {
                if( m_nanopt == KEEP )
                    m_lastval = std::numeric_limits<double>::quiet_NaN();
            }
            else
            {
                m_lastval = x;
                if( m_method == MAX )
                    m_maxtree.insert( x );
                else
                    m_mintree.insert( x );
            }
        }

        void remove( double x )
        {
            if( likely( !isnan( x ) ) )
            {
                if ( m_method == MAX )
                    m_maxtree.erase ( m_maxtree.find( x ) );
                else
                    m_mintree.erase ( m_mintree.find( x ) );
            }
        }

        void reset()
        {
            if( m_method == MAX )
                m_maxtree.clear();
            else
                m_mintree.clear();
        }

        double compute() const
        {
            // Verify tree is not empty and lastValue is valid
            // Last value can only ever be NaN if the "keep" nan option is used
            if( likely( !isnan( m_lastval ) && ( ( m_method == MAX && m_maxtree.size() > 0 ) || m_mintree.size() > 0 ) ) )
            {
                switch( m_method )
                {
                    case MIN:
                    {
                        if ( m_mintree.size() == 1 )
                            return 0;
                        return m_mintree.get<0>().find_rank( m_lastval );
                    }
                    case MAX:
                    {
                        if ( m_maxtree.size() == 1 )
                            return 0;
                        return m_maxtree.size() - 1 - m_maxtree.get<0>().find_rank( m_lastval );
                    }
                    case AVG:
                    {
                        if ( m_mintree.size() == 1 )
                            return 0;
                        
                        int min_rank = m_mintree.get<0>().find_rank( m_lastval );
                        int max_rank = min_rank;
                        auto it = m_mintree.get<0>().nth( min_rank );
                        it++;
                        for( ; it != m_mintree.end() && *it == m_lastval ; it++ ) max_rank++; // While this is in theory O(n), in reality this loop is only interated once, since there are likely no duplicate values or very few.
                        return ( double )( min_rank + max_rank ) / 2;
                    }
                    default:
                        break;
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        }

    private:
        ost<std::less<double>> m_mintree;
        ost<std::greater<double>> m_maxtree;
        double m_lastval = std::numeric_limits<double>::quiet_NaN();

        int64_t m_method;
        int64_t m_nanopt;
};

class ArgMinMax
{
    public:
        ArgMinMax() = default;
        ArgMinMax( bool max, bool recent ) : m_recent( recent ), m_monoQueue( max ) {}

        ArgMinMax( ArgMinMax && rhs ) = default;
        ArgMinMax & operator=( ArgMinMax && rhs ) = default;

        // no copy
        ArgMinMax( const ArgMinMax & rhs ) = delete;
        ArgMinMax & operator=( const ArgMinMax & rhs ) = delete;

        void add( double x, DateTime t )
        {
            m_monoQueue.add( x );
            auto & it = m_treemap[x];
            it.m_count++;
            if( m_recent )
                it.m_lasttime = t;
            else
                it.m_alltimes.push( t );
        }

        void remove( double x )
        {
            m_monoQueue.remove( x );
            auto it = m_treemap.find( x );
            it->second.m_count--;
            if( !it->second.m_count ) // don't let map grow unbounded
                m_treemap.erase( it );
            else if( !m_recent )
                it->second.m_alltimes.pop_left();
        }

        void reset()
        {
            m_monoQueue.reset();
            m_treemap.clear();
        }

        DateTime compute()
        {
            if( m_treemap.size() > 0 )
            {
                double arg_val = m_monoQueue.compute();
                if( m_recent )
                    return m_treemap[arg_val].m_lasttime;
                else
                    return m_treemap[arg_val].m_alltimes[-1];
            }

            return DateTime::fromNanoseconds( 0 );
        }

    private:
        struct TreeData
        {
            TreeData & operator=( const TreeData & rhs ) = delete;
            TreeData & operator=( TreeData && rhs ) = default;

            int m_count = 0;
            DateTime m_lasttime;
            VariableSizeWindowBuffer<DateTime> m_alltimes;
        };

        bool m_recent;
        AscendingMinima m_monoQueue;
        std::map<double, TreeData> m_treemap;
};

class EMA
{
    public:
        EMA() = default;

        EMA( double alpha, bool ignore_na, int64_t horizon, bool adjust )
        {
            m_alpha = alpha;
            m_ignore_na = ignore_na;
            reset();
        }

        EMA( EMA && rhs ) = default;

        EMA & operator=( EMA && rhs ) = default;

        void add( double x )
        {
            if( unlikely( m_first ) && !isnan( x ) )
            {
                m_ema = x;
                m_first = false;
            }
            else if( unlikely( isnan( x ) ) && !m_ignore_na && likely( !m_first ) )
            {
                m_offset++;
            }
            else if( likely( !isnan( x ) ) )
            {
                double delta = x - m_ema;
                if( m_offset == 1 )
                {
                    m_ema += m_alpha * delta;
                }
                else
                {
                    m_ema = ( m_ema * pow( ( 1 - m_alpha ), m_offset ) + m_alpha * x ) / 
                        ( pow( 1 - m_alpha, m_offset ) + m_alpha );
                    m_offset = 1;
                }
            }
        }

        void remove( double x ) { }

        void reset()
        {
            m_ema = 0;
            m_offset = 1;
            m_first = true;
        }

        double compute() const
        {
            return unlikely( m_first ) ? std::numeric_limits<double>::quiet_NaN() : m_ema;
        }

    private:

        double m_ema;
        int64_t m_offset;
        bool m_first;

        double m_alpha;
        bool m_ignore_na;
};

class AdjustedEMA
{
    public:
        AdjustedEMA() = default;

        AdjustedEMA( double alpha, bool ignore_na, int64_t horizon, bool adjust )
        {
            m_decay = 1 - alpha;
            m_ignore_na = ignore_na;
            m_horizon = horizon;
            reset();
        }

        AdjustedEMA( AdjustedEMA && rhs ) = default;

        AdjustedEMA & operator=( AdjustedEMA && rhs ) = default;

        void add( double x )
        {
            if( likely( !isnan( x ) ) )
            {
                double decay_factor = ( m_ignore_na ? m_decay : pow( m_decay, m_offset ) );
                m_ema *= decay_factor;
                m_norm *= decay_factor;
                m_offset = 1;
                m_ema += x;
                m_norm++;
            }
            else
            {
                m_offset++;
                m_nan_count++;
            }
        }

        void remove( double x )
        {
            if( likely( !isnan( x ) ) )
            {
                double lookback = ( m_ignore_na ? m_horizon - m_nan_count : m_horizon - m_offset + 1 );
                double decay_factor = pow( m_decay, lookback );
                m_norm -= decay_factor;
                m_ema -= x * decay_factor;

                // EMA may go to zero with a non-zero normalizer due to lots of 0 values in the time series: they are separate conditions
                if( abs( m_ema ) < EPSILON )
                    m_ema = 0;
            }
            else
                m_nan_count--;
        }

        void reset()
        {
            m_ema = m_norm = m_nan_count = 0;
            m_offset = 1;
        }

        double compute() const
        {
            if( m_norm > 0 ) // may be 0 if calc is triggered before any ticks are received
                return m_ema / m_norm;

            return std::numeric_limits<double>::quiet_NaN();
        }

    private:

        double m_ema;
        double m_norm;
        int64_t m_offset;
        double m_nan_count;

        double m_decay;
        bool m_ignore_na;
        int64_t m_horizon;
};

class AlphaDebiasEMA
{
    public:
        AlphaDebiasEMA() = default;

        AlphaDebiasEMA( double alpha, bool ignore_na, int64_t horizon, bool adjust )
        {
            m_decay = 1 - alpha;
            m_ignore_na = ignore_na;
            m_horizon = horizon;
            m_adjust = adjust;
            reset();
        }
        
        void add( double x )
        {
            if( likely( !isnan( x ) ) )
            {
                if( unlikely( m_first ) )
                {
                    m_wsum = 1;
                    m_sqsum = 1;
                    m_first = false;
                }
                else
                {
                    double decay_factor = ( m_ignore_na ? m_decay : pow( m_decay, m_offset ) );
                    m_wsum *= decay_factor;
                    m_sqsum *= decay_factor * decay_factor;
                    m_offset = 1;

                    double w0;
                    if( m_adjust )
                        w0 = 1.0;
                    else
                        w0 = 1 - m_decay;
                    m_sqsum += w0 * w0;
                    m_wsum += w0;
                    if( !m_adjust )
                    {
                        double correction = decay_factor + w0;
                        m_wsum /= correction;
                        m_sqsum /= ( correction * correction );
                    }
                }
            }
            else
            {
                if( likely( !m_first ) )
                    m_offset++;
                m_nan_count++;
            }
        }

        void remove( double x )
        {
            if( likely( !isnan( x ) ))
            {
                double lookback = ( m_ignore_na ? m_horizon - m_nan_count : m_horizon - m_offset + 1 );
                double wh = pow( m_decay, lookback );
                if( !m_adjust )
                    wh *= ( 1- m_decay );
                m_sqsum -= wh * wh;
                m_wsum -= wh;
                if( m_wsum < EPSILON || m_sqsum < EPSILON )
                {
                    m_wsum = 0;
                    m_sqsum = 0;
                }
            }
            else
                m_nan_count--;
        }

        void reset()
        {
            m_wsum = m_sqsum = m_nan_count = 0;
            m_offset = 1;
            m_first = true;
        }

        double compute() const
        {
            double wsum_sq = m_wsum * m_wsum;
            if( abs( wsum_sq - m_sqsum ) > EPSILON )
                return wsum_sq / ( wsum_sq - m_sqsum );
            else
                return std::numeric_limits<double>::quiet_NaN();
        }

    private:

        double m_wsum;
        double m_sqsum;
        int64_t m_offset;
        double m_nan_count;
        bool m_first;

        double m_decay;
        int64_t m_horizon;
        bool m_ignore_na, m_adjust;
};

class HalflifeEMA
{
public:
    HalflifeEMA() = default;

    HalflifeEMA( TimeDelta halflife, DateTime start, bool )
    {
        m_decay_factor = log( 0.5 ) / halflife.asNanoseconds();
        reset();
    }

    void add( double x, DateTime now )
    {
        if( unlikely( m_last_tick.isNone() ) )
            m_ema = x;
        else
        {
            double decay = 1 - exp( m_decay_factor * ( now - m_last_tick ).asNanoseconds() );
            m_ema += decay * ( x - m_ema );
        }
        m_last_tick = now;
    }

    void reset()
    {
        m_ema = std::numeric_limits<double>::quiet_NaN();
        m_last_tick = DateTime::NONE();
    }

    double compute() const
    {
        return m_ema;
    }

private:
    double    m_ema;
    double    m_decay_factor;
    DateTime  m_last_tick;
};

class AdjustedHalflifeEMA
{
    public:
        AdjustedHalflifeEMA() = default;

        AdjustedHalflifeEMA( TimeDelta halflife, DateTime start, bool )
        {
            m_decay_factor = log( 0.5 ) / halflife.asNanoseconds();
            m_last_tick = start;
            reset();
        }

        void add( double x, DateTime now )
        {
            TimeDelta delta_t = now - m_last_tick;
            double decay = exp( m_decay_factor * delta_t.asNanoseconds() );
            m_ema = decay * m_ema + x;
            m_norm = decay * m_norm + 1.0;
            m_last_tick = now;
        }

        void reset()
        {
            m_ema = m_norm = 0;
        }

        double compute() const
        {
            return likely( m_norm > 0 ) ? ( m_ema / m_norm ) : std::numeric_limits<double>::quiet_NaN();
        }

    private:

        double m_ema;
        double m_norm;
        double m_decay_factor;
        DateTime m_last_tick;
};

class HalflifeDebiasEMA
{
    public:
        HalflifeDebiasEMA() = default;

        HalflifeDebiasEMA( TimeDelta halflife, DateTime start, bool adjust )
        {
            m_decay = log( 0.5 ) / halflife.asNanoseconds();
            m_last_tick = start;
            m_adjust = adjust;
            reset();
        }

        void add( double x, DateTime now )
        {
            TimeDelta delta_t = now - m_last_tick;
            double decay_factor = exp( m_decay * delta_t.asNanoseconds() );
            m_sqsum *= decay_factor * decay_factor;
            m_wsum *= decay_factor;

            double w0;
            if( m_adjust )
                w0 = 1.0;
            else
                w0 = 1 - m_decay;
            m_sqsum += w0 * w0;
            m_wsum += w0;

            m_last_tick = now;
        }

        void reset()
        {
            m_wsum = m_sqsum = 0;
        }

        double compute() const
        {
            double wsum_sq = m_wsum * m_wsum;
            if( wsum_sq != m_sqsum )
                return wsum_sq / ( wsum_sq - m_sqsum );
            else
                return std::numeric_limits<double>::quiet_NaN();
        }

    private:

        double m_wsum;
        double m_sqsum;
        double m_decay;
        bool   m_adjust;
        DateTime m_last_tick;
};

// NaN handling generic code for the DataValidator class
struct NanCheck
{
    template <typename FloatingT, typename ...V, std::enable_if_t<std::is_floating_point<FloatingT>::value, bool> = true>
    static inline bool any_nan( FloatingT val, V... args )
    {
        return isnan( val ) || any_nan( args... );
    }

    template <typename NonFloatingT, typename ...V, std::enable_if_t<!std::is_floating_point<NonFloatingT>::value, bool> = false>
    static inline bool any_nan( NonFloatingT val, V... args )
    {
        return any_nan( args... );
    }

    static inline bool any_nan()
    {
        return false;
    }
};

template <typename T, typename... Types>
struct is_any_of : std::false_type {};

template <typename T, typename First, typename... Rest>
struct is_any_of<T, First, Rest...> { static constexpr bool value = std::is_same_v<T, First> || is_any_of<T, Rest...>::value; };

template <typename T, typename... Types>
inline constexpr bool is_any_of_v = is_any_of<T, Types...>::value;

// Validates min_data_points and takes care of NaN handling
template<typename T>
class DataValidator
{
    static constexpr bool PROCESS_NA    = is_any_of_v<T, EMA, AdjustedEMA, AlphaDebiasEMA, Rank>;
    static constexpr bool CONSIDER_NA   = is_any_of_v<T, First, Last>;

    public:
        DataValidator() = default;

        DataValidator( DataValidator && rhs ) = default;

        DataValidator( const DataValidator & rhs ) = default;

        DataValidator & operator=( DataValidator && rhs ) = default;

        DataValidator & operator=( const DataValidator & rhs ) = default;

        template<typename ...V>
        DataValidator( int64_t min_data_points, bool ignore_na, V... args ) :
            m_mdp( min_data_points ), m_igna( ignore_na ), m_stat ( args... ) { }

        void add( double x )
        {
            if( isnan( x ) )
            {
                m_nans++;
                if( PROCESS_NA || ( CONSIDER_NA && !m_igna ) )
                    m_stat.add( x );
            }
            else
            {
                m_points++;
                m_stat.add( x );
            }
        }

        template<typename ...V>
        void add( V... args )
        {
            if( NanCheck::any_nan( args...) )
                m_nans++;
            else
            {
                m_points++;
                m_stat.add( args... );
            }
        }

        template<typename ...V>
        void remove( V... args )
        {
            if( NanCheck::any_nan( args...) )
            {
                m_nans--;
                if( PROCESS_NA || ( CONSIDER_NA && !m_igna ) )
                    m_stat.remove( args... );
            }
            else
            {
                m_points--;
                m_stat.remove( args... );
            }
        }

        template<typename ...V>
        double compute( V... args )
        {
            if( ( !m_igna && (m_nans > 0  && !CONSIDER_NA ) ) || m_points < m_mdp )
                return std::numeric_limits<double>::quiet_NaN();

            return m_stat.compute( args... );
        }

        DateTime compute_dt() // needed for argmin/max
        {
            if( ( !m_igna && m_nans > 0 ) || m_points < m_mdp )
                return DateTime::fromNanoseconds( 0 );

            return m_stat.compute();
        }

        void reset()
        {
            m_nans = 0;
            m_points = 0;
            m_stat.reset();
        }

    private:
        int64_t m_nans = 0;
        int64_t m_points = 0;
        int64_t m_mdp = 0;
        bool m_igna = false;
        T m_stat;
};


// Generic window update nodes

template<typename T, typename NodeT>
DECLARE_CPPNODE( _generic_tick_window_updates )
{
protected:
    TS_INPUT( T, x );
    SCALAR_INPUT( int64_t, interval );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, sampler );
    TS_INPUT( Generic, reset );
    TS_INPUT( Generic, recalc );

    STATE_VAR( bool, s_first{true} );
    STATE_VAR( int64_t, s_last_call{0} );
    STATE_VAR( int64_t, s_last_count{0} );
    STATE_VAR( bool, s_pending_recalc{false} );

    STATE_VAR( FixedSizeWindowBuffer<T>, s_value_buffer{interval} );
    STATE_VAR( std::vector<T>, s_removals{} );

    TS_NAMED_OUTPUT_RENAMED( std::vector<T>, additions, additions_ );
    TS_NAMED_OUTPUT_RENAMED( std::vector<T>, removals, removals_ );

    CSP csp;
    const char * name() const override { return "_tick_window_updates"; }

public:
    _generic_tick_window_updates( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    START()
    {
        if( interval <= 0 )
            CSP_THROW( ValueError, "Tick interval needs to be positive" );
        csp.make_passive( x );
    }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            // clear the buffer
            s_value_buffer.clear();
            s_removals.clear();
            s_last_count = 0;
        }

        if( csp.ticked( recalc ) )
        {
            s_pending_recalc = true;
        }

        if( csp.ticked( sampler ) )
        {
            // Handle removals if needed
            if( s_value_buffer.full() && s_removals.size() < ( size_t )s_last_count )
            {
                T v = s_value_buffer.pop_left();
                s_removals.push_back( v );
            }

            NodeT * node = static_cast<NodeT*>( this ); // CRTP
            if( csp.ticked( x ) )
            {
                // Ensure size is consistent for NumPy case - do nothing for floats
                node -> validateShape();
                s_value_buffer.push( x );
            }
            else
            {
                node -> checkValid();
                s_value_buffer.push( node -> createNan() );
            }
        }

        if( csp.ticked( trigger ) || ( s_first && csp.ticked( x ) && csp.ticked( sampler )  ) )
        {
            s_first = false;
            int64_t total_count = csp.count( sampler );

            // Handle removals
            if( !s_removals.empty() && !s_pending_recalc )
            {
                std::swap( removals_.reserveSpace(), s_removals );
                s_removals.clear();
            }

            if( s_pending_recalc && !s_value_buffer.empty() )
            {
                // Copy entire buffer to additions for fresh calculation
                std::vector<T>* additions = &additions_.reserveSpace();
                additions -> reserve( s_value_buffer.count() );
                s_value_buffer.copy_values( additions );
                s_pending_recalc = false;
            }
            else
            {
                // Handle incremental additions since last call
                size_t ticks = total_count - s_last_call;
                size_t sz = s_value_buffer.count();
                int64_t add_cutoff = std::min( ticks, sz );
                int64_t offset = add_cutoff-1;

                std::vector<T>* additions = nullptr;
                while( offset >= 0 )
                {
                    if( !additions ) // reserve memory
                    {
                        additions = &additions_.reserveSpace();
                        additions -> clear();
                    }
                    additions -> push_back( s_value_buffer[offset] );
                    offset--;
                }
            }

            s_last_call = total_count;
            s_last_count = s_value_buffer.count(); // needed for no improper removals
        }
    }
};

template<typename T, typename NodeT>
DECLARE_CPPNODE( _generic_time_window_updates )
{
protected:

    TS_INPUT( T, x );
    SCALAR_INPUT( TimeDelta, interval );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, sampler );
    TS_INPUT( Generic, reset );
    TS_INPUT( Generic, recalc );

    STATE_VAR( DateTime, s_last_call );
    STATE_VAR( bool, s_pending_recalc{false} );
    STATE_VAR( int64_t, s_last_tick{0} );

    STATE_VAR( size_t,  s_pending_removals{0} );
    // This keeps track of how many additions have been published without corresponding removals
    // Its to fix a rare bug when trigger is separate from the data, and the data has multiple ticks at the same timestamp
    // In this case the first data tick gets picked up on trigger and sent with additions, but the second tick does NOT
    // get picked up, because trigger already finished.  Then when its time to remove, the value that wasnt sent on additions
    // gets added to removed because its past the timestamp... example:
    // import csp
    // from datetime import datetime, timedelta
    // import csp.stats
    // data = [
    //   (datetime(2023, 2, 17,17, 17, 34), 23.0),
    //   (datetime(2023, 2, 17,17, 22, 43), 22.0),
    //   (datetime(2023, 2, 17,17, 22, 43), 22.0),
    //   (datetime(2023, 2, 17,17, 23, 25), 22.0),
    //   (datetime(2023, 2, 17,17, 50, 45), 21.0),
    //   (datetime(2023, 2, 17,18, 20, 49), 21.0),
    //   (datetime(2023, 2, 17,18, 21, 0), 22.0),
    // ]
    // trigger_data=[
    //   (datetime(2023, 2, 17, 17, 22, 43), True ),
    //   (datetime(2023, 2, 17, 18, 21, 0), True ),
    //   (datetime(2023, 2, 17, 18, 22, 0), True ),
    // ]
    // def g():
    //   d = csp.curve(float, data)
    //   trigger = csp.curve(bool, trigger_data)
    //   m = csp.stats.max(d, timedelta(minutes=31), min_window=timedelta(), trigger=trigger)
    //   csp.print('trigger', trigger)
    //   csp.print('d', d)
    //   csp.print('m', m)
    // csp.run(g, starttime=datetime(2023,2,17,12), endtime=datetime(2023, 2, 17, 18, 25))
    //
    // We use a simple counter to work around this issue, we wont output a removal if there are no pending remaining

    STATE_VAR( bool, s_first{true} );
    STATE_VAR( bool, s_expanding{false} );

    STATE_VAR( VariableSizeWindowBuffer<T>, s_value_buffer{} );
    STATE_VAR( VariableSizeWindowBuffer<DateTime>, s_time_buffer{} );

    STATE_VAR( std::vector<T>, s_additions{} ); // only used for expanding window optimization

    TS_NAMED_OUTPUT_RENAMED( std::vector<T>, additions, additions_ );
    TS_NAMED_OUTPUT_RENAMED( std::vector<T>, removals, removals_ );

    CSP csp;
    const char * name() const override { return "_time_window_updates"; }

public:

    _generic_time_window_updates( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    START()
    {
        if( TimeDelta( interval ) < TimeDelta::ZERO() )
            CSP_THROW( ValueError, "Time interval needs to be non-negative" );
        else if( TimeDelta( interval ) == TimeDelta::ZERO() )
            s_expanding = true;
        csp.make_passive( x );
    }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            if( s_expanding )
                s_additions.clear();
            else
            {
                s_value_buffer.clear();
                s_time_buffer.clear();
                s_pending_removals = 0;
            }
        }

        if( csp.ticked( recalc ) )
        {
            s_pending_recalc = true;
        }

        if( csp.ticked( sampler ) )
        {
            NodeT * node = static_cast<NodeT*>( this ); // CRTP
            if( csp.ticked( x ) )
            {
                node -> validateShape();
                if( s_expanding )
                    s_additions.push_back( x );
                else
                    s_value_buffer.push( x );
            }
            else
            {
                node -> checkValid();
                if( s_expanding )
                    s_additions.push_back( node -> createNan() );
                else
                    s_value_buffer.push( node -> createNan() );
            }
            if( !s_expanding )
                s_time_buffer.push( now() );
        }

        if( csp.ticked( trigger ) || ( s_first && csp.ticked( x ) && csp.ticked( sampler ) ) )
        {
            s_first = false;

            if( s_expanding )
            {
                // fast track expanding window calculations
                // we just need to swap in all additions with no checks 
                if( !s_additions.empty() )
                {
                    std::swap( additions_.reserveSpace(), s_additions );
                    s_additions.clear(); // keep allocated memory
                }
                return;
            }
            
            DateTime threshold = now() - interval;
            // Handle removals
            int64_t count = csp.count( sampler );
            std::vector<T>* removals = nullptr;
            while( !s_value_buffer.empty() && s_time_buffer[-1] <= threshold )
            {
                DateTime time = s_time_buffer.pop_left();
                T val = s_value_buffer.pop_left();

                if( !s_pending_recalc && time <= s_last_call && s_pending_removals )
                {
                    if( !removals ) // reserve memory
                    {
                        removals = &removals_.reserveSpace();
                        removals -> clear();
                    }
                    removals -> push_back( val );
                    --s_pending_removals;
                }
            }

            // Handle additions
            if( s_pending_recalc && !s_value_buffer.empty() )
            {
                // Copy entire buffer to additions for fresh calculation
                std::vector<T>* additions = &additions_.reserveSpace();
                additions -> reserve( s_value_buffer.count() );
                s_value_buffer.copy_values( additions );
                s_pending_recalc = false;
                s_pending_removals = additions -> size();
            }
            else
            {
                // Handle additions incrementally
                int64_t sz = s_time_buffer.count();
                int64_t offset = std::min( count - s_last_tick, sz ) - 1;

                std::vector<T>* additions = nullptr;
                while( offset >= 0 )
                {
                    if( !additions ) // reserve memory
                    {
                        additions = &additions_.reserveSpace();
                        additions -> clear();
                    }
                    additions -> push_back( s_value_buffer[offset] );
                    offset--;
                    ++s_pending_removals;
                }
            }

            s_last_call = now();
            s_last_tick = count;
        }
    }
};

template<typename T, typename OutputT, typename NodeT>
DECLARE_CPPNODE( _generic_cross_sectional )
{
protected:

    TS_INPUT( std::vector<T>, additions );
    TS_INPUT( std::vector<T>, removals );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    STATE_VAR( VariableSizeWindowBuffer<T>, s_window{} );
    TS_OUTPUT( OutputT );

    CSP csp;
    const char * name() const override { return "_cross_sectional"; }

public:

    _generic_cross_sectional( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    INVOKE()
    {
        if( csp.ticked( reset ) )
            s_window.clear();
        if( csp.ticked( removals ) )
            s_window.remove_left( removals.lastValue().size() );
        if( csp.ticked( additions ) )
            s_window.extend( additions.lastValue() );
        if( csp.ticked( trigger ) )
            static_cast<NodeT*>( this ) -> computeCrossSectional(); // CRTP
    }
};

}

#endif // _IN_CSP_CPPNODES_STATSIMPL_H
