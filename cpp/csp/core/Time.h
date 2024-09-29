#ifndef _IN_CSP_CORE_TIME_H
#define _IN_CSP_CORE_TIME_H

#include <csp/core/Exception.h>
#include <csp/core/Platform.h>
#include <csp/core/System.h>
#include <math.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <limits>
#include <string>

namespace csp
{
const int64_t NANOS_PER_MICROSECOND = 1000;
const int64_t NANOS_PER_MILLISECOND = 1000000;
const int64_t NANOS_PER_SECOND      = 1000000000;
const int64_t SECONDS_PER_DAY       = 86400;
const int64_t NANOS_PER_DAY         = NANOS_PER_SECOND * SECONDS_PER_DAY;

class TimeDelta
{
public:
    constexpr TimeDelta() : TimeDelta( TimeDelta::NONE() ) {}
    constexpr TimeDelta( int64_t seconds, int64_t nanoseconds ) : TimeDelta( seconds * NANOS_PER_SECOND + nanoseconds )
    {}

    int64_t asNanoseconds() const  { return m_ticks; }
    int64_t asMicroseconds() const { return m_ticks / NANOS_PER_MICROSECOND; }
    int64_t asMilliseconds() const { return m_ticks / NANOS_PER_MILLISECOND; }
    int64_t asSeconds() const      { return m_ticks / NANOS_PER_SECOND; }

    int32_t days() const         { return asSeconds() / SECONDS_PER_DAY; }
    int32_t hours() const        { return ( asSeconds() % SECONDS_PER_DAY ) / 3600; }
    int32_t minutes() const      { return ( asSeconds() % 3600 ) / 60; }
    int32_t seconds() const      { return asSeconds() % 60; }
    
    int32_t nanoseconds() const  { return asNanoseconds() % NANOS_PER_SECOND; }

    TimeDelta abs() const        { return TimeDelta( std::abs( m_ticks ) ); }
    int sign() const {
        if(m_ticks < 0) {
            return -1;
        } else if (m_ticks > 0) {
            return 1;
        }
        return 0;
    }

    bool isNone() const          { return (*this) == TimeDelta::NONE(); }

    std::string asString() const;
    
    //from XX units
    static constexpr TimeDelta fromNanoseconds(  int64_t nanos )   { return TimeDelta( nanos ); }
    static constexpr TimeDelta fromMicroseconds( int64_t micros )  { return fromNanoseconds( micros  * NANOS_PER_MICROSECOND ); }
    static constexpr TimeDelta fromMilliseconds( int64_t millis )  { return fromNanoseconds( millis  * NANOS_PER_MILLISECOND ); }
    static constexpr TimeDelta fromSeconds(      int64_t seconds ) { return fromNanoseconds( seconds * NANOS_PER_SECOND ); }
    static constexpr TimeDelta fromMinutes(      int64_t minutes ) { return fromSeconds( minutes * 60 ); }
    static constexpr TimeDelta fromHours(        int64_t hours )   { return fromSeconds( hours * 3600 ); }
    static constexpr TimeDelta fromDays(         int64_t days )    { return fromSeconds( days * SECONDS_PER_DAY ); }

    //HH:MM:SS.nn format, see Time::fromString
    static TimeDelta fromString( const std::string & str );

    bool operator==( const TimeDelta & rhs ) const { return m_ticks == rhs.m_ticks; }
    bool operator!=( const TimeDelta & rhs ) const { return !( (*this) == rhs ); }

    bool operator< ( const TimeDelta & rhs ) const  { return m_ticks <  rhs.m_ticks; }
    bool operator<=( const TimeDelta & rhs ) const  { return m_ticks <= rhs.m_ticks; }
    bool operator> ( const TimeDelta & rhs ) const  { return m_ticks >  rhs.m_ticks; }
    bool operator>=( const TimeDelta & rhs ) const  { return m_ticks >= rhs.m_ticks; }

    TimeDelta operator +( const TimeDelta & rhs ) const { return TimeDelta( m_ticks + rhs.m_ticks ); }
    TimeDelta operator -( const TimeDelta & rhs ) const { return TimeDelta( m_ticks - rhs.m_ticks ); }
    TimeDelta & operator+=( const TimeDelta & rhs )     { m_ticks += rhs.m_ticks; return *this; }
    TimeDelta & operator-=( const TimeDelta & rhs )     { m_ticks -= rhs.m_ticks; return *this; }

    TimeDelta operator*( int mult ) const { return TimeDelta( m_ticks * mult ); }
    TimeDelta & operator*=( int mult )    { m_ticks *= mult; return *this; }

    TimeDelta operator/( int div ) const  { return TimeDelta( m_ticks / div ); }
    TimeDelta & operator/=( int div )     { m_ticks /= div; return *this; }

    int64_t operator/( TimeDelta div ) const { return m_ticks / div.m_ticks; }

    TimeDelta operator-() const       { return TimeDelta( -m_ticks ); }

    static constexpr TimeDelta ZERO()      { return TimeDelta( 0 ); }
    static constexpr TimeDelta NONE()      { return TimeDelta( std::numeric_limits<int64_t>::min() ); }
    static constexpr TimeDelta MIN_VALUE() { return TimeDelta( std::numeric_limits<int64_t>::min() + 1 ); } //min reserved for NONE
    static constexpr TimeDelta MAX_VALUE() { return TimeDelta( std::numeric_limits<int64_t>::max() ); }

private:
    //the fact that we store this as nanos is an implementation detail
    constexpr TimeDelta( int64_t raw_nanos ) : m_ticks( raw_nanos ) {}
    int64_t m_ticks;
};  

inline std::string TimeDelta::asString() const
{
    char buf[64];

    int32_t d = days();
    int32_t h = hours();
    int32_t m = minutes();
    int32_t s = seconds();
    int32_t n = nanoseconds();

    int idx = d ? sprintf( buf, "%d %s ", d, d == 1 ? "day" : "days" ) : 0;
    idx += sprintf( buf + idx, "%02d:%02d:%02d", h, m, s );
    if( n )
        sprintf( buf + idx, ".%09d", n );
    return buf;
}

inline TimeDelta TimeDelta::fromString( const std::string & str )
{
    //can do S, as seconds
    //M:S for minutes:seconds
    //H:M:S for hours:minutes:seconds
    //all version can take .nnn for fractions of a second
    int h = 0;
    int m = 0;
    int s = 0;
    int v1 = 0;
    int v2 = 0;
    int v3 = 0;
    int ns = 0;
    const char * c_str = str.c_str();
    int n = sscanf( c_str, "%d:%d:%d", &v1, &v2, &v3 );
    if( n == 0 )
        CSP_THROW( ValueError, "Failed to convert " << str << " to TimeDelta" );

    if( n == 1 )
        s = v1;
    else if( n == 2 )
    {
        s = v2;
        m = v1;
    }
    else if( n != EOF )
    {
        s = v3;
        m = v2;
        h = v1;
    }
    
    char * frac = strrchr( ( char * ) c_str, '.' );
    if( frac )
    {
        ns = atoi( frac + 1 );
        int len = strlen( frac + 1 );
        ns *= pow( 10, 9 - len );
    }
    return TimeDelta( NANOS_PER_SECOND * ( h * 3600 + m * 60 + s ) + ns );
}

inline std::ostream & operator <<( std::ostream &os, const TimeDelta & d )
{
    os << d.asString();
    return os;
}

class Date
{
public:
    Date() : Date( NONE() ) {}
    Date( int16_t year, int8_t month, int8_t day );

    int16_t year() const  { return repr().year;  }
    int8_t  month() const { return repr().month; }
    int8_t  day() const   { return repr().day;   }

    bool isNone() const   { return (*this) == Date::NONE(); }

    size_t strftime( char *result, size_t max_size, const char *fmt ) const;

    bool operator==( const Date & rhs ) const { return m_data.value == rhs.m_data.value; }
    bool operator!=( const Date & rhs ) const { return !( (*this) == rhs ); }
    
    bool operator< ( const Date & rhs ) const  { return m_data.value <  rhs.m_data.value; }
    bool operator<=( const Date & rhs ) const  { return m_data.value <= rhs.m_data.value; }
    bool operator> ( const Date & rhs ) const  { return m_data.value >  rhs.m_data.value; }
    bool operator>=( const Date & rhs ) const  { return m_data.value >= rhs.m_data.value; }

    Date operator-( const TimeDelta & delta ) const;
    Date operator+( const TimeDelta & delta ) const;
    TimeDelta operator-( const Date & rhs ) const;

    Date & operator-=( const TimeDelta & delta ) { *this = *this - delta; return *this; }
    Date & operator+=( const TimeDelta & delta ) { *this = *this + delta; return *this; }

    //day of week, 0 = Sunday, 6 = Saturday
    int weekday() const     { return asTM().tm_wday; }
    bool isWeekday() const  { return !isWeekend(); }
    bool isWeekend() const;

    std::string asString() const   { return asYYYYMMDD(); }
    std::string asYYYYMMDD() const;

    size_t hash() const { return std::hash<int32_t>()( m_data.value ); }

    static Date today();
    static Date fromYYYYMMDD( const std::string & );

    static Date NONE() { return Date( -1, -1, -1 ); }

private:
    Date( const tm & TM ) : Date( TM.tm_year + 1900, TM.tm_mon + 1, TM.tm_mday ) {}

    tm asTM( bool do_mktime = true ) const;

    //ordering here is important! year is placed at the high order bits
    //for correct comparisons against m_value
    struct _repr
    {
        int8_t  day;
        int8_t  month;
        int16_t year;
    };

    static_assert( sizeof( _repr ) == sizeof( int32_t ) );

    union data
    {
        _repr   repr;
        int32_t value;
    };

    const _repr & repr() const { return m_data.repr; }
    _repr & repr()             { return m_data.repr; }

    data m_data; //first 16 bits = year, next 8 = month, next 8 = day
};

inline Date::Date( int16_t year, int8_t month, int8_t day )
{
    repr().year  = year;
    repr().month = month;
    repr().day   = day;
}

inline tm Date::asTM( bool do_mktime ) const
{
    tm TM{0};
    TM.tm_year  = year() - 1900;
    TM.tm_mon   = month() - 1;
    TM.tm_mday   = day();
    TM.tm_isdst = -1;
    if( do_mktime )
        mktime( &TM );
    return TM;
}

inline Date Date::operator-( const TimeDelta & delta ) const
{
    tm TM = asTM( false );
    TM.tm_mday -= delta.days();
    mktime( &TM );
    return Date( TM );
}

inline Date Date::operator+( const TimeDelta & delta ) const
{
    tm TM = asTM( false );
    TM.tm_mday += delta.days();
    mktime( &TM );
    return Date( TM );
}

inline bool Date::isWeekend() const
{
    int d = weekday();
    return d == 0 || d == 6;
}

inline size_t Date::strftime( char *result, size_t max_size, const char *fmt ) const
{
    tm time = asTM();
    return ::strftime( result, max_size, fmt, &time );
}

inline Date Date::fromYYYYMMDD( const std::string & date )
{
    int year;
    int month;
    int day;
    if( sscanf( date.c_str(), "%04d%02d%02d", &year, &month, &day ) != 3 )
        CSP_THROW( InvalidArgument, "Date string not in YYYYMMDD format: " << date );

    return Date( year, month, day );
}

inline Date Date::today()
{
    tm TM;
    time_t t = time( NULL );
    localtime_r( &t, &TM );
    return Date( TM );
}

inline std::string Date::asYYYYMMDD() const 
{
    char buf[32];
    sprintf( buf, "%04d%02d%02d", year(), month(), day() );
    return buf;
}

inline std::ostream & operator <<( std::ostream &os, const Date & d )
{
    os << d.asString();
    return os;
}

class Time
{
public:
    Time() : Time( -1 ) {} //NONE
    Time( int hour, int minute, int second, int32_t nanosecond = 0 );

    int     hour() const       { return asSeconds() / 3600; }
    int     minute() const     { return ( asSeconds() % 3600 ) / 60; }
    int     second() const     { return asSeconds() % 60; }
    int32_t nanosecond() const { return m_ticks % NANOS_PER_SECOND; }

    bool isNone() const        { return (*this) == Time::NONE(); }

    //from XX since midnight
    static Time fromNanoseconds( int64_t nanos )   { return Time( nanos ); }
    static Time fromMicroseconds( int64_t micros ) { return fromNanoseconds( micros * NANOS_PER_MICROSECOND ); }
    static Time fromMilliseconds( int64_t millis ) { return fromNanoseconds( millis * NANOS_PER_MILLISECOND ); }
    static Time fromSeconds( int64_t seconds )     { return fromNanoseconds( seconds * NANOS_PER_SECOND ); }

    //as XX units since midnight
    int64_t asNanoseconds() const  { return m_ticks; }
    int64_t asMicroseconds() const { return m_ticks / NANOS_PER_MICROSECOND; }
    int64_t asMilliseconds() const { return m_ticks / NANOS_PER_MILLISECOND; }
    int64_t asSeconds() const      { return m_ticks / NANOS_PER_SECOND; }

    bool operator==( const Time & rhs ) const { return m_ticks == rhs.m_ticks; }
    bool operator!=( const Time & rhs ) const { return !( (*this) == rhs ); }
    
    bool operator< ( const Time & rhs ) const  { return m_ticks <  rhs.m_ticks; }
    bool operator<=( const Time & rhs ) const  { return m_ticks <= rhs.m_ticks; }
    bool operator> ( const Time & rhs ) const  { return m_ticks >  rhs.m_ticks; }
    bool operator>=( const Time & rhs ) const  { return m_ticks >= rhs.m_ticks; }

    Time operator +( const TimeDelta & delta ) const { return Time( m_ticks + delta.asNanoseconds() ); }
    Time operator -( const TimeDelta & delta ) const { return Time( m_ticks - delta.asNanoseconds() ); }
    TimeDelta operator -( const Time & rhs ) const   { return TimeDelta::fromNanoseconds( m_ticks - rhs.m_ticks ); }

    Time& operator +=( const TimeDelta & delta );
    Time& operator -=( const TimeDelta & delta );

    std::string asString() const;
    static Time fromString( const std::string & );

    static Time NONE() { return Time( -1 ); }
    static Time MIN_VALUE()  { return Time( 0, 0, 0 ); }

private:
    Time( int64_t raw );

    void checkRange( int64_t t );

    //stored as nanos since midnight
    int64_t m_ticks;
};

inline Time::Time( int64_t raw )
{
    checkRange( raw );
    m_ticks = raw;
}

inline void Time::checkRange( int64_t raw )
{
    if( raw >= SECONDS_PER_DAY * NANOS_PER_SECOND || raw < -1 )
        CSP_THROW( ValueError, "Time value out of range: " << raw );
}

inline Time::Time( int hour, int minute, int second, int32_t nanosecond )
{
    if( hour > 23 || hour < 0 )
        CSP_THROW( ValueError, "Hour out of range: " << hour );
    if( minute > 59 || minute < 0 )
        CSP_THROW( ValueError, "Minute out of range: " << minute );
    if( second > 59 || second < 0 )
        CSP_THROW( ValueError, "Second out of range: " << second );
    if( nanosecond >= NANOS_PER_SECOND || nanosecond < 0 )
        CSP_THROW( ValueError, "Nanosecond out of range: " << nanosecond );

    m_ticks = ( int64_t( hour ) * 3600 + int64_t( minute ) * 60 + int64_t( second ) ) * NANOS_PER_SECOND + nanosecond;
}

inline Time& Time::operator +=( const TimeDelta & delta )
{
    int64_t newval = m_ticks + delta.asNanoseconds(); 
    checkRange( newval );
    m_ticks = newval;
    return *this; 
}

inline Time& Time::operator -=( const TimeDelta & delta ) 
{
    int64_t newval = m_ticks - delta.asNanoseconds(); 
    checkRange( newval );
    m_ticks = newval;
    return *this; 
}

inline std::string Time::asString() const
{
    char buf[64];
    sprintf( buf, "%02d:%02d:%02d.%09d", hour(), minute(), second(), nanosecond() );
    return buf;
}

inline Time Time::fromString( const std::string & str )
{
    int h = 0;
    int m = 0;
    int s = 0;
    char f[16] = "";
    int ns = 0;
    int n = sscanf( str.c_str(), "%d:%d:%d.%s", &h, &m, &s, f );
    if( n == 0 )
        CSP_THROW( ValueError, "Failed to convert " << str << " to Time" );

    char *end;
    ns = strtol( f, &end, 10 );
    int len = end - f;
    ns *= pow( 10, 9 - len );
    return Time( h, m, s, ns );
}

inline std::ostream & operator <<( std::ostream &os, const Time & t )
{
    os << t.asString();
    return os;
}

// Time is internally stored as an int64_t nanoseconds since 1970. 
// All DateTime objects are stored as UTC and should be treated as such
class DateTime
{
public:
    DateTime() : DateTime( DateTime::NONE() ) {}
    DateTime( int year, int month, int day,
              int hour = 0, int minute = 0, int second = 0, int nanosecond = 0 );
    DateTime( Date date, Time time );

    //Note this returns a shared thread-local buffer, invalidated on next call on same thread
    const char * asCString() const;
    const char * asCString( char * buf, size_t buflen ) const;
    std::string asString() const { return asCString(); }

    //Helper creation methods
    static DateTime now();

    static DateTime fromString( const std::string & dtstr );

    //from XX units since epoch
    static DateTime fromNanoseconds(  int64_t nanos )   { return DateTime( nanos ); }
    static DateTime fromMicroseconds( int64_t micros )  { return fromNanoseconds( micros  * NANOS_PER_MICROSECOND ); }
    static DateTime fromMilliseconds( int64_t millis )  { return fromNanoseconds( millis  * NANOS_PER_MILLISECOND ); }
    static DateTime fromSeconds(      int64_t seconds ) { return fromNanoseconds( seconds * NANOS_PER_SECOND ); }

    //as XX units of time since epoch
    int64_t asNanoseconds() const  { return m_ticks; }
    int64_t asMicroseconds() const { return m_ticks / NANOS_PER_MICROSECOND; }
    int64_t asMilliseconds() const { return m_ticks / NANOS_PER_MILLISECOND; }
    int64_t asSeconds() const      { return m_ticks / NANOS_PER_SECOND; }

    bool isNone() const            { return (*this) == DateTime::NONE(); }
    bool isMin() const             { return (*this) == DateTime::MIN_VALUE(); }
    bool isMax() const             { return (*this) == DateTime::MAX_VALUE(); }

    //returns time / date component of the datetime
    Time time() const { return isNone() ? Time::NONE() : Time::fromNanoseconds( m_ticks % ( SECONDS_PER_DAY * NANOS_PER_SECOND ) ); }
    Date date() const;

    //return a datetime with same date but with given time
    DateTime withTime( Time t ) const { return DateTime( ( m_ticks / ( SECONDS_PER_DAY * NANOS_PER_SECOND ) ) * SECONDS_PER_DAY * NANOS_PER_SECOND + t.asNanoseconds() ); }

    //round to given timedelta ( commonly used in bucketing )
    DateTime roundDown( TimeDelta td ) const { return DateTime::fromNanoseconds( ( m_ticks / td.asNanoseconds() ) * td.asNanoseconds() ); }

    bool operator==( const DateTime & rhs ) const { return m_ticks == rhs.m_ticks; }
    bool operator!=( const DateTime & rhs ) const { return !( (*this) == rhs ); }
    
    bool operator< ( const DateTime & rhs ) const  { return m_ticks <  rhs.m_ticks; }
    bool operator<=( const DateTime & rhs ) const  { return m_ticks <= rhs.m_ticks; }
    bool operator> ( const DateTime & rhs ) const  { return m_ticks >  rhs.m_ticks; }
    bool operator>=( const DateTime & rhs ) const  { return m_ticks >= rhs.m_ticks; }
    
    DateTime operator +( const TimeDelta & delta ) const { return DateTime( m_ticks + delta.asNanoseconds() ); }
    DateTime operator -( const TimeDelta & delta ) const { return DateTime( m_ticks - delta.asNanoseconds() ); }
    TimeDelta operator -( const DateTime & rhs ) const   { return TimeDelta::fromNanoseconds( m_ticks - rhs.m_ticks ); }

    DateTime& operator +=( const TimeDelta & delta ) { m_ticks += delta.asNanoseconds(); return *this; }
    DateTime& operator -=( const TimeDelta & delta ) { m_ticks -= delta.asNanoseconds(); return *this; }

    static constexpr DateTime NONE()      { return DateTime(std::numeric_limits<int64_t>::min()); }
    static constexpr DateTime MIN_VALUE() { return DateTime( std::numeric_limits<int64_t>::min() + 1 ); }  //min reserved for NONE
    static constexpr DateTime MAX_VALUE() { return DateTime( std::numeric_limits<int64_t>::max() ); }

protected:
    //the fact that we store this as nanos is an implementation detail
    constexpr DateTime( int64_t raw_nanos ) : m_ticks( raw_nanos ) {}
    tm asTM() const;

    int64_t m_ticks;
};

inline DateTime::DateTime( int year, int month, int day,
                           int hour, int minute, int second, int nanosecond )
{
    tm TM;
    memset( &TM, 0, sizeof( TM ) );
    TM.tm_year = year - 1900;
    TM.tm_mon  = month - 1;
    TM.tm_mday = day;
    TM.tm_hour = hour;
    TM.tm_min  = minute;
    TM.tm_sec  = second;
    TM.tm_isdst = -1;

    m_ticks = timegm( &TM );
    m_ticks = m_ticks * NANOS_PER_SECOND + nanosecond;

}

inline DateTime::DateTime( Date date, Time time ) :
    DateTime( date.year(), date.month(), date.day(), time.hour(), time.minute(), time.second(), time.nanosecond() )
{
}

inline DateTime DateTime::now()
{
    timespec ts;
#ifdef WIN32
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime( CLOCK_REALTIME, &ts );
#endif
    return DateTime( ts.tv_sec * NANOS_PER_SECOND + ts.tv_nsec );
}

inline DateTime DateTime::fromString( const std::string & str )
{
    return { Date::fromYYYYMMDD( str ), Time::fromString( str.c_str() + 9 ) };
}


inline const char * DateTime::asCString() const
{
    static thread_local char s_buf[128];
    return asCString( s_buf, sizeof( s_buf ) );
}

inline const char * DateTime::asCString( char * buf, size_t buflen ) const
{
    if( (*this) == DateTime::NONE() )
        return strncpy( buf, "none", buflen );

    if( (*this) == DateTime::MIN_VALUE() )
        return strncpy( buf, "min", buflen );

    if( (*this) == DateTime::MAX_VALUE() )
        return strncpy( buf, "max", buflen );
    
    tm TM = asTM();

    size_t len;
    if( ( len = strftime( buf, buflen, "%Y%m%d %H:%M:%S", &TM ) ) == 0 )
        CSP_THROW( RuntimeException, "strftime failed" );

    auto nanos = m_ticks % NANOS_PER_SECOND;
    if( nanos < 0 )
        nanos += NANOS_PER_SECOND;

    snprintf( buf + len, buflen - len, ".%09ld", (long int) nanos );
    return buf;
}

inline std::ostream & operator <<( std::ostream &os, const DateTime & dt )
{
    os << dt.asString();
    return os;
}

//Helper class to extract day/month/year/etc info from raw timestamp
//ie DateTimeEx dte( existingDt )
//dte.day, etc etc
class DateTimeEx : public DateTime
{
public:
    DateTimeEx( const DateTime & dt );

    int day() const   { return m_tm.tm_mday; }
    int month() const { return m_tm.tm_mon + 1; }
    int year() const  { return m_tm.tm_year + 1900; }
    
    int hour() const   { return m_tm.tm_hour; }
    int minute() const { return m_tm.tm_min; }
    int second() const { return m_tm.tm_sec; }

    //the fractional second access are non-cumulative, meaning microseconds() includes milliseconds().
    //ie if we have micros 222333, milliseconds() wil return 222 and microseconds will return 222333
    int milliseconds() const { return nanoseconds() / NANOS_PER_MILLISECOND; }
    int microseconds() const { return nanoseconds() / NANOS_PER_MICROSECOND; }
    int nanoseconds() const  
    {
        auto nanos = m_ticks % NANOS_PER_SECOND;
        if( unlikely( nanos < 0 ) )
           nanos += NANOS_PER_SECOND;
        return nanos;
    }
 
    //day of week, 0 = Sunday, 6 = Saturday
    int weekday() const { return m_tm.tm_wday; }

private:
    tm m_tm;
};

inline DateTimeEx::DateTimeEx( const DateTime & dt ) : DateTime( dt )
{
    m_tm = asTM();
}

inline Date DateTime::date() const
{
    DateTimeEx dtEx( *this );
    return Date( dtEx.year(), dtEx.month(), dtEx.day() );
}

inline void sleep( TimeDelta delta )
{
    timespec ts;
    ts.tv_sec  = delta.asSeconds();
    ts.tv_nsec = delta.nanoseconds();
    nanosleep( &ts, NULL );
}

inline TimeDelta Date::operator-( const Date & rhs ) const
{
    return DateTime( *this, Time( 0, 0, 0 ) ) - DateTime( rhs, Time( 0, 0, 0 ) );
}

};

//hash definition for unordered_set / unordered_map keys
namespace std
{
    template<> struct hash<csp::DateTime>
    {
        size_t operator()( const csp::DateTime & dt ) const
        {
            return std::hash< int64_t >()( static_cast<int64_t>( dt.asNanoseconds() ) );
        }
    };

    template<> struct hash<csp::Date>
    {
        size_t operator()( const csp::Date & dt ) const
        {
            return dt.hash();
        }
    };

    template<> struct hash<csp::Time>
    {
        size_t operator()( const csp::Time & t ) const
        {
            return std::hash< int64_t >()( static_cast<int64_t>( t.asNanoseconds() ) );
        }
    };

    template<> struct hash<csp::TimeDelta>
    {
        size_t operator()( const csp::TimeDelta & td ) const
        {
            return std::hash< int64_t >()( static_cast<int64_t>( td.asNanoseconds() ) );
        }
    };
}
#endif
