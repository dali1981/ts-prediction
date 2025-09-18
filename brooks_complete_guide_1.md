# Al Brooks Complete Price Action Trading Guide

## Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Fundamental Concepts](#fundamental-concepts)
3. [Bar Analysis](#bar-analysis)
4. [Market Structure](#market-structure)
5. [Trading Setups](#trading-setups)
6. [Risk Management](#risk-management)
7. [Trading Guidelines](#trading-guidelines)
8. [Python Implementation](#python-implementation)

---

## Core Philosophy

Al Brooks' approach is based on **reading price charts bar by bar** without traditional indicators. His methodology focuses on:

- **Pure Price Action**: No indicators, just raw price movement
- **Always-In Direction**: Market is always either long or short
- **Probability Over Prediction**: Focus on setups with statistical edge
- **Bar-by-Bar Analysis**: Every bar tells a story about supply and demand

### Key Principle: The Trader's Equation
> **Only take trades where: (Probability of Success × Reward) > (Probability of Failure × Risk)**

---

## Fundamental Concepts

### 1. Always-In Direction
The market is **ALWAYS** in either a bull or bear trend. Brooks' rule:
- **Bull Market**: If 7 of the last 10 bars are mostly above the EMA
- **Bear Market**: If 7 of the last 10 bars are mostly below the EMA
- **Never trade against the Always-In direction**

### 2. Two-Legged Moves
- Every move tends to have two legs
- Every correction tends to have two legs
- This fractal nature repeats on all timeframes

### 3. Measured Moves
- **Leg 1 = Leg 2**: Most common measured move
- Markets often move in symmetrical legs
- Use height of first leg to project target for second leg

### 4. Gaps as Strength Indicators
- **Breakout Gap**: Start of new trend
- **Measuring Gap**: Middle of trend, projects measured move
- **Exhaustion Gap**: End of trend, potential reversal

---

## Bar Analysis

### Signal Bars

#### Bull Reversal Bar
- Close above open (bull body)
- Close above midpoint
- Lower tail about 1/3 to 1/2 of bar height
- Small or no upper tail
- Little overlap with prior bars

#### Bear Reversal Bar
- Close below open (bear body) 
- Close below midpoint
- Upper tail about 1/3 to 1/2 of bar height
- Small or no lower tail
- Little overlap with prior bars

### Trend Bars

#### Bull Trend Bar
- Close near high
- Open near low
- Small upper tail (<30% of body)
- Body >70% of range

#### Bear Trend Bar
- Close near low
- Open near high  
- Small lower tail (<30% of body)
- Body >70% of range

### Special Bars

#### Doji Bar
- Body <30% of range
- Indicates indecision
- Often marks end of leg

#### Inside Bar (ii pattern)
- High ≤ prior bar high
- Low ≥ prior bar low
- Breakout setup

#### Outside Bar
- High > prior bar high
- Low < prior bar low
- Often reversal signal

---

## Market Structure

### Higher Highs and Lower Lows

#### Bull Trend Structure
- **Higher High (HH)**: Swing high > previous swing high
- **Higher Low (HL)**: Swing low > previous swing low
- Trend intact as long as HH and HL continue

#### Bear Trend Structure  
- **Lower Low (LL)**: Swing low < previous swing low
- **Lower High (LH)**: Swing high < previous swing high
- Trend intact as long as LL and LH continue

### Swing Points
- **Swing High**: Bar with high > highs of bars on both sides (typically 3 bars)
- **Swing Low**: Bar with low < lows of bars on both sides

---

## Trading Setups

### 1. Bar Counting (High 1, High 2, Low 1, Low 2)

#### High 1, High 2 Pattern (Bull Setup)
- **High 1**: First bar with high > prior bar high in pullback
- **High 2**: Second bar with high > prior bar high
- **Entry**: Buy stop above High 2 bar
- **Stop**: Below signal bar low
- **Target**: Measured move or 2:1 minimum

#### Low 1, Low 2 Pattern (Bear Setup)
- **Low 1**: First bar with low < prior bar low in rally
- **Low 2**: Second bar with low < prior bar low  
- **Entry**: Sell stop below Low 2 bar
- **Stop**: Above signal bar high
- **Target**: Measured move or 2:1 minimum

### 2. Second Entries (Highest Probability)

Brooks emphasizes second entries as the **best setups**:

#### Bull Second Entry
- Failed first attempt to reverse down
- Higher low after initial pullback
- Buy above second signal bar
- **Success rate: 60-70%**

#### Bear Second Entry
- Failed first attempt to reverse up
- Lower high after initial rally
- Sell below second signal bar
- **Success rate: 60-70%**

### 3. Wedge Patterns (Three-Push)

#### Rising Wedge (Bearish)
- Three consecutive higher highs
- Declining momentum on each push
- Often at resistance
- Short below third push

#### Falling Wedge (Bullish)
- Three consecutive lower lows
- Declining momentum on each push
- Often at support
- Buy above third push

### 4. Spike and Channel

Most common daily pattern:

#### Bull Spike and Channel
1. **Spike**: Strong vertical move up (2-5 bars)
2. **Channel**: Slower grind higher
3. **Test**: Pullback to channel bottom
4. **Double Bottom**: Buy at channel start retest

#### Bear Spike and Channel
1. **Spike**: Strong vertical move down (2-5 bars)
2. **Channel**: Slower grind lower
3. **Test**: Rally to channel top
4. **Double Top**: Sell at channel start retest

### 5. Failed Breakouts

High probability reversal setups:

#### Failed Bull Breakout
- Break above resistance
- Immediate reversal back below
- Short below failure bar
- **Success rate: 70-80%**

#### Failed Bear Breakout
- Break below support
- Immediate reversal back above
- Buy above failure bar
- **Success rate: 70-80%**

### 6. Double Tops and Bottoms

#### Double Bottom Bull Flag
- Two tests of same low level
- In context of bull trend
- Buy above second bottom
- Target: Measured move up

#### Double Top Bear Flag
- Two tests of same high level
- In context of bear trend
- Sell below second top
- Target: Measured move down

### 7. Final Flags

#### Bull Final Flag
- Last push up in exhausted trend
- Often wedge-shaped
- Followed by two-legged correction
- Minimum target: Bottom of flag

#### Bear Final Flag
- Last push down in exhausted trend
- Often wedge-shaped
- Followed by two-legged correction
- Minimum target: Top of flag

---

## Risk Management

### Position Sizing
- **Risk per trade**: 2% of account maximum
- **Beginner risk**: 0.5-1% until profitable

### Stop Placement
- **Initial stop**: Beyond signal bar extreme
- **After entry bar closes**: Move to entry bar extreme if trend bar
- **Breakeven**: After minimum scalp profit achieved

### Profit Targets

#### Scalping (Not Recommended for Beginners)
- **Target**: 1-3 points (Emini)
- **Risk**: 2 points
- **Required win rate**: >70%
- **Brooks says**: "You will not make consistent money until you stop trading countertrend scalps"

#### Swing Trading (Recommended)
- **Minimum target**: Equal to risk (1:1)
- **Preferred target**: 2× risk (2:1)
- **Optimal**: Measured move or major S/R
- **Required win rate**: >40%

### The Mathematics of Trading

For Emini with 10-15 point daily range:
- **Standard risk**: 2 points
- **Minimum profit**: 2 points
- **Swing target**: 4+ points

**Critical Rule**: Never risk more than your reward unless probability >80%

---

## Trading Guidelines

### Brooks' 10 Most Important Rules

1. **Trade with-trend pullbacks only** until consistently profitable
2. **Wait for second entries** - first attempts usually fail
3. **Never trade countertrend** in strong trends
4. **Risk = Reward minimum** for all trades
5. **Discipline is everything** - follow rules without exception
6. **Do nothing for hours** if no good setups
7. **Increase position size**, not trade frequency
8. **Two points per day** in Emini is enough (50 contracts = 7 figures/year)
9. **Print charts daily** and mark every setup
10. **If you lost last month**, don't trade reversals

### Entry Techniques

#### Stop Entry (Most Common)
- Buy stop above signal bar high
- Sell stop below signal bar low
- Used for breakouts and with-trend

#### Limit Entry (Advanced)
- Buy at/below prior bar low in bull trend
- Sell at/above prior bar high in bear trend
- Requires strong trend context

### Trade Management

#### Scaling In
- Add to winners, not losers
- Same position size each add
- Only in strong trends

#### Scaling Out
- Exit 50% at minimum target
- Exit 25% at 2× target
- Hold 25% for runner

---

## Python Implementation

### Complete Al Brooks Trading System

```python
"""
Al Brooks Price Action Trading System
Complete Implementation with All Patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class MarketStructure(Enum):
    """Market structure types"""
    BULL_TREND = "Bull Trend"
    BEAR_TREND = "Bear Trend"
    TRADING_RANGE = "Trading Range"
    BREAKOUT_MODE = "Breakout Mode"


class SetupType(Enum):
    """All Brooks setup types"""
    HIGH_1 = "High 1"
    HIGH_2 = "High 2"
    HIGH_3 = "High 3 (Wedge)"
    HIGH_4 = "High 4"
    LOW_1 = "Low 1"
    LOW_2 = "Low 2"
    LOW_3 = "Low 3 (Wedge)"
    LOW_4 = "Low 4"
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    WEDGE_TOP = "Wedge Top"
    WEDGE_BOTTOM = "Wedge Bottom"
    SPIKE_AND_CHANNEL_BULL = "Spike and Channel Bull"
    SPIKE_AND_CHANNEL_BEAR = "Spike and Channel Bear"
    FAILED_BREAKOUT_BULL = "Failed Bull Breakout"
    FAILED_BREAKOUT_BEAR = "Failed Bear Breakout"
    FINAL_FLAG_BULL = "Final Flag Bull"
    FINAL_FLAG_BEAR = "Final Flag Bear"
    SECOND_ENTRY_BULL = "Second Entry Bull"
    SECOND_ENTRY_BEAR = "Second Entry Bear"
    MEASURING_GAP = "Measuring Gap"
    EXHAUSTION_GAP = "Exhaustion Gap"


@dataclass
class BarAnalysis:
    """Complete bar analysis"""
    index: int
    timestamp: pd.Timestamp
    bar_type: str
    is_bull_bar: bool
    is_bear_bar: bool
    is_doji: bool
    is_inside: bool
    is_outside: bool
    is_reversal: bool
    body_percent: float
    upper_tail_percent: float
    lower_tail_percent: float
    overlap_percent: float
    strength: float  # 0-1 score


@dataclass
class Setup:
    """Trading setup with all details"""
    timestamp: pd.Timestamp
    type: SetupType
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    target1: float  # Minimum target
    target2: float  # Measured move
    probability: float  # Estimated success rate
    risk_reward: float
    strength: float  # Setup quality 0-1
    notes: str
    context: Dict = field(default_factory=dict)


class AlBrooksPriceAction:
    """
    Complete Al Brooks Price Action Trading System
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 ema_period: int = 20,
                 swing_strength: int = 3):
        """
        Initialize Brooks trading system
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        ema_period : int
            Period for EMA (Brooks uses 20)
        swing_strength : int
            Bars on each side for swing identification
        """
        self.data = data.copy()
        self.ema_period = ema_period
        self.swing_strength = swing_strength
        self.setups = []
        
        self._prepare_data()
        self._identify_market_structure()
        self._analyze_bars()
        
    def _prepare_data(self):
        """Calculate all technical data"""
        df = self.data
        
        # Price metrics
        df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        df['atr'] = self._calculate_atr()
        
        # Bar measurements
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['body_midpoint'] = (df['open'] + df['close']) / 2
        df['bar_midpoint'] = (df['high'] + df['low']) / 2
        
        # Tails
        df['upper_tail'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_tail'] = df[['close', 'open']].min(axis=1) - df['low']
        
        # Bar relationships
        df['bull_bar'] = df['close'] > df['open']
        df['bear_bar'] = df['close'] < df['open']
        df['doji'] = df['body'] / df['range'] < 0.3
        
        # Swing points
        df['swing_high'] = self._find_swing_highs()
        df['swing_low'] = self._find_swing_lows()
        
        # Trend structure
        df['hh'] = False  # Higher high
        df['ll'] = False  # Lower low
        df['hl'] = False  # Higher low
        df['lh'] = False  # Lower high
        self._identify_trend_structure()
        
    def _calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        df = self.data
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
        
    def _find_swing_highs(self) -> pd.Series:
        """Identify swing highs"""
        highs = pd.Series(False, index=self.data.index)
        for i in range(self.swing_strength, len(self.data) - self.swing_strength):
            window = self.data['high'].iloc[i-self.swing_strength:i+self.swing_strength+1]
            if self.data['high'].iloc[i] == window.max():
                highs.iloc[i] = True
        return highs
        
    def _find_swing_lows(self) -> pd.Series:
        """Identify swing lows"""
        lows = pd.Series(False, index=self.data.index)
        for i in range(self.swing_strength, len(self.data) - self.swing_strength):
            window = self.data['low'].iloc[i-self.swing_strength:i+self.swing_strength+1]
            if self.data['low'].iloc[i] == window.min():
                lows.iloc[i] = True
        return lows
        
    def _identify_trend_structure(self):
        """Identify HH, LL, HL, LH"""
        df = self.data
        
        # Get swing points
        swing_highs = df[df['swing_high']].copy()
        swing_lows = df[df['swing_low']].copy()
        
        # Higher highs and lower highs
        for i in range(1, len(swing_highs)):
            curr_idx = swing_highs.index[i]
            prev_idx = swing_highs.index[i-1]
            
            if swing_highs['high'].iloc[i] > swing_highs['high'].iloc[i-1]:
                df.loc[curr_idx, 'hh'] = True
            else:
                df.loc[curr_idx, 'lh'] = True
                
        # Lower lows and higher lows
        for i in range(1, len(swing_lows)):
            curr_idx = swing_lows.index[i]
            prev_idx = swing_lows.index[i-1]
            
            if swing_lows['low'].iloc[i] < swing_lows['low'].iloc[i-1]:
                df.loc[curr_idx, 'll'] = True
            else:
                df.loc[curr_idx, 'hl'] = True
                
    def _identify_market_structure(self):
        """Determine market structure for each bar"""
        df = self.data
        df['market_structure'] = MarketStructure.TRADING_RANGE.value
        
        for i in range(20, len(df)):
            # Count bars above/below EMA (Brooks' Always-In rule)
            last_10 = df.iloc[i-10:i]
            above_ema = (last_10['close'] > last_10['ema']).sum()
            below_ema = (last_10['close'] < last_10['ema']).sum()
            
            # Recent trend structure
            recent_hh = df['hh'].iloc[i-20:i].sum()
            recent_ll = df['ll'].iloc[i-20:i].sum()
            recent_hl = df['hl'].iloc[i-20:i].sum()
            recent_lh = df['lh'].iloc[i-20:i].sum()
            
            # Determine structure
            if above_ema >= 7 and recent_hh > recent_lh:
                df.loc[df.index[i], 'market_structure'] = MarketStructure.BULL_TREND.value
            elif below_ema >= 7 and recent_ll > recent_hl:
                df.loc[df.index[i], 'market_structure'] = MarketStructure.BEAR_TREND.value
            elif df['range'].iloc[i-5:i].mean() < df['atr'].iloc[i] * 0.5:
                df.loc[df.index[i], 'market_structure'] = MarketStructure.BREAKOUT_MODE.value
                
    def _analyze_bars(self):
        """Detailed analysis of each bar"""
        analyses = []
        df = self.data
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Calculate percentages
            body_pct = curr['body'] / curr['range'] if curr['range'] > 0 else 0
            upper_tail_pct = curr['upper_tail'] / curr['range'] if curr['range'] > 0 else 0
            lower_tail_pct = curr['lower_tail'] / curr['range'] if curr['range'] > 0 else 0
            
            # Overlap with previous bar
            overlap_high = min(curr['high'], prev['high'])
            overlap_low = max(curr['low'], prev['low'])
            overlap = max(0, overlap_high - overlap_low)
            overlap_pct = overlap / prev['range'] if prev['range'] > 0 else 0
            
            # Determine bar type
            bar_type = self._classify_bar_type(i)
            
            # Calculate strength
            strength = self._calculate_bar_strength(i)
            
            analysis = BarAnalysis(
                index=i,
                timestamp=df.index[i],
                bar_type=bar_type,
                is_bull_bar=curr['bull_bar'],
                is_bear_bar=curr['bear_bar'],
                is_doji=curr['doji'],
                is_inside=curr['high'] <= prev['high'] and curr['low'] >= prev['low'],
                is_outside=curr['high'] > prev['high'] and curr['low'] < prev['low'],
                is_reversal=self._is_reversal_bar(i),
                body_percent=body_pct,
                upper_tail_percent=upper_tail_pct,
                lower_tail_percent=lower_tail_pct,
                overlap_percent=overlap_pct,
                strength=strength
            )
            analyses.append(analysis)
            
        self.bar_analyses = analyses
        
    def _classify_bar_type(self, idx: int) -> str:
        """Classify bar according to Brooks' definitions"""
        curr = self.data.iloc[idx]
        
        if curr['doji']:
            return "Doji"
        
        if curr['bull_bar']:
            if curr['upper_tail'] < curr['body'] * 0.3:
                return "Bull Trend Bar"
            elif curr['lower_tail'] > curr['body'] * 1.5:
                return "Bull Reversal Bar"
            else:
                return "Bull Bar"
        else:
            if curr['lower_tail'] < curr['body'] * 0.3:
                return "Bear Trend Bar"
            elif curr['upper_tail'] > curr['body'] * 1.5:
                return "Bear Reversal Bar"
            else:
                return "Bear Bar"
                
    def _is_reversal_bar(self, idx: int) -> bool:
        """Check if bar is a reversal bar"""
        if idx < 2:
            return False
            
        curr = self.data.iloc[idx]
        prev = self.data.iloc[idx-1]
        prev2 = self.data.iloc[idx-2]
        
        # Bull reversal
        if (curr['bull_bar'] and 
            prev['bear_bar'] and
            curr['close'] > prev['high'] and
            curr['lower_tail'] > curr['body'] * 0.5):
            return True
            
        # Bear reversal
        if (curr['bear_bar'] and
            prev['bull_bar'] and
            curr['close'] < prev['low'] and
            curr['upper_tail'] > curr['body'] * 0.5):
            return True
            
        return False
        
    def _calculate_bar_strength(self, idx: int) -> float:
        """Calculate bar strength (0-1)"""
        curr = self.data.iloc[idx]
        strength = 0.0
        
        # Body size relative to range
        body_ratio = curr['body'] / curr['range'] if curr['range'] > 0 else 0
        strength += body_ratio * 0.3
        
        # Close location
        if curr['bull_bar']:
            close_location = (curr['close'] - curr['low']) / curr['range'] if curr['range'] > 0 else 0
        else:
            close_location = (curr['high'] - curr['close']) / curr['range'] if curr['range'] > 0 else 0
        strength += close_location * 0.3
        
        # Range relative to ATR
        range_ratio = min(1.0, curr['range'] / curr['atr']) if curr['atr'] > 0 else 0
        strength += range_ratio * 0.2
        
        # Trend alignment
        if curr['market_structure'] == MarketStructure.BULL_TREND.value and curr['bull_bar']:
            strength += 0.2
        elif curr['market_structure'] == MarketStructure.BEAR_TREND.value and curr['bear_bar']:
            strength += 0.2
            
        return min(1.0, strength)
        
    def find_high_low_patterns(self) -> List[Setup]:
        """Find High 1, High 2, Low 1, Low 2 patterns"""
        setups = []
        df = self.data
        
        for i in range(30, len(df) - 2):
            # Skip if not in trend
            if df['market_structure'].iloc[i] == MarketStructure.TRADING_RANGE.value:
                continue
                
            # Bull patterns (High 1, High 2)
            if df['market_structure'].iloc[i] == MarketStructure.BULL_TREND.value:
                pattern = self._find_high_pattern(i)
                if pattern:
                    setups.append(pattern)
                    
            # Bear patterns (Low 1, Low 2)
            elif df['market_structure'].iloc[i] == MarketStructure.BEAR_TREND.value:
                pattern = self._find_low_pattern(i)
                if pattern:
                    setups.append(pattern)
                    
        return setups
        
    def _find_high_pattern(self, idx: int) -> Optional[Setup]:
        """Find High 1, High 2, High 3 patterns in pullback"""
        df = self.data
        
        # Look for pullback
        high_count = 0
        pattern_bars = []
        
        for i in range(idx - 20, idx):
            if df['high'].iloc[i] > df['high'].iloc[i-1]:
                high_count += 1
                pattern_bars.append(i)
                
                if high_count == 2:  # High 2 found
                    # Check for valid setup
                    signal_bar = df.iloc[i]
                    
                    # Need trend line break between High 1 and High 2
                    if not self._has_trend_line_break(pattern_bars[0], i):
                        continue
                        
                    # Calculate setup
                    entry = signal_bar['high'] + 0.01
                    stop = signal_bar['low'] - 0.01
                    risk = entry - stop
                    target1 = entry + risk  # 1:1
                    target2 = entry + (risk * 2)  # 2:1
                    
                    # Probability based on pattern type
                    if high_count == 2:
                        setup_type = SetupType.HIGH_2
                        probability = 0.60
                    else:
                        setup_type = SetupType.HIGH_3
                        probability = 0.65
                        
                    return Setup(
                        timestamp=df.index[i],
                        type=setup_type,
                        direction="LONG",
                        entry_price=entry,
                        stop_loss=stop,
                        target1=target1,
                        target2=target2,
                        probability=probability,
                        risk_reward=2.0,
                        strength=self._calculate_setup_strength(i, "LONG"),
                        notes=f"{setup_type.value} in bull trend",
                        context={"pattern_bars": pattern_bars}
                    )
                    
        return None
        
    def _find_low_pattern(self, idx: int) -> Optional[Setup]:
        """Find Low 1, Low 2, Low 3 patterns in rally"""
        df = self.data
        
        # Look for rally in bear trend
        low_count = 0
        pattern_bars = []
        
        for i in range(idx - 20, idx):
            if df['low'].iloc[i] < df['low'].iloc[i-1]:
                low_count += 1
                pattern_bars.append(i)
                
                if low_count == 2:  # Low 2 found
                    # Check for valid setup
                    signal_bar = df.iloc[i]
                    
                    # Need trend line break between Low 1 and Low 2
                    if not self._has_trend_line_break(pattern_bars[0], i):
                        continue
                        
                    # Calculate setup
                    entry = signal_bar['low'] - 0.01
                    stop = signal_bar['high'] + 0.01
                    risk = stop - entry
                    target1 = entry - risk  # 1:1
                    target2 = entry - (risk * 2)  # 2:1
                    
                    # Probability based on pattern type
                    if low_count == 2:
                        setup_type = SetupType.LOW_2
                        probability = 0.60
                    else:
                        setup_type = SetupType.LOW_3
                        probability = 0.65
                        
                    return Setup(
                        timestamp=df.index[i],
                        type=setup_type,
                        direction="SHORT",
                        entry_price=entry,
                        stop_loss=stop,
                        target1=target1,
                        target2=target2,
                        probability=probability,
                        risk_reward=2.0,
                        strength=self._calculate_setup_strength(i, "SHORT"),
                        notes=f"{setup_type.value} in bear trend",
                        context={"pattern_bars": pattern_bars}
                    )
                    
        return None
        
    def _has_trend_line_break(self, start_idx: int, end_idx: int) -> bool:
        """Check for trend line break between two points"""
        if end_idx - start_idx < 2:
            return False
            
        df = self.data
        
        # Simple check: any bar crossed EMA
        for i in range(start_idx + 1, end_idx):
            if df['low'].iloc[i] < df['ema'].iloc[i] < df['high'].iloc[i]:
                return True
                
        return False
        
    def find_wedge_patterns(self) -> List[Setup]:
        """Find wedge (three-push) patterns"""
        setups = []
        df = self.data
        
        for i in range(50, len(df) - 5):
            # Rising wedge (bearish)
            rising = self._find_rising_wedge(i)
            if rising:
                setups.append(rising)
                
            # Falling wedge (bullish)
            falling = self._find_falling_wedge(i)
            if falling:
                setups.append(falling)
                
        return setups
        
    def _find_rising_wedge(self, idx: int) -> Optional[Setup]:
        """Find rising wedge pattern"""
        df = self.data
        lookback = 30
        
        # Find three swing highs
        swing_highs = []
        for i in range(idx - lookback, idx):
            if df['swing_high'].iloc[i]:
                swing_highs.append(i)
                
        if len(swing_highs) < 3:
            return None
            
        # Check for three higher highs
        highs = [df['high'].iloc[i] for i in swing_highs[-3:]]
        if not (highs[2] > highs[1] > highs[0]):
            return None
            
        # Check for momentum divergence
        momentum = []
        for i in range(1, len(swing_highs)):
            move = abs(df['high'].iloc[swing_highs[i]] - df['high'].iloc[swing_highs[i-1]])
            time = swing_highs[i] - swing_highs[i-1]
            momentum.append(move / time if time > 0 else 0)
            
        if len(momentum) < 2 or momentum[-1] >= momentum[0]:
            return None  # No divergence
            
        # Setup found
        signal_idx = idx
        entry = df['low'].iloc[signal_idx] - 0.01
        stop = df['high'].iloc[swing_highs[-1]] + 0.01
        risk = stop - entry
        target1 = entry - risk
        target2 = entry - (risk * 2)
        
        return Setup(
            timestamp=df.index[signal_idx],
            type=SetupType.WEDGE_TOP,
            direction="SHORT",
            entry_price=entry,
            stop_loss=stop,
            target1=target1,
            target2=target2,
            probability=0.65,
            risk_reward=2.0,
            strength=self._calculate_setup_strength(signal_idx, "SHORT"),
            notes="Rising wedge with momentum divergence",
            context={"swing_highs": swing_highs, "momentum": momentum}
        )
        
    def _find_falling_wedge(self, idx: int) -> Optional[Setup]:
        """Find falling wedge pattern"""
        df = self.data
        lookback = 30
        
        # Find three swing lows
        swing_lows = []
        for i in range(idx - lookback, idx):
            if df['swing_low'].iloc[i]:
                swing_lows.append(i)
                
        if len(swing_lows) < 3:
            return None
            
        # Check for three lower lows
        lows = [df['low'].iloc[i] for i in swing_lows[-3:]]
        if not (lows[2] < lows[1] < lows[0]):
            return None
            
        # Check for momentum divergence
        momentum = []
        for i in range(1, len(swing_lows)):
            move = abs(df['low'].iloc[swing_lows[i]] - df['low'].iloc[swing_lows[i-1]])
            time = swing_lows[i] - swing_lows[i-1]
            momentum.append(move / time if time > 0 else 0)
            
        if len(momentum) < 2 or momentum[-1] >= momentum[0]:
            return None  # No divergence
            
        # Setup found
        signal_idx = idx
        entry = df['high'].iloc[signal_idx] + 0.01
        stop = df['low'].iloc[swing_lows[-1]] - 0.01
        risk = entry - stop
        target1 = entry + risk
        target2 = entry + (risk * 2)
        
        return Setup(
            timestamp=df.index[signal_idx],
            type=SetupType.WEDGE_BOTTOM,
            direction="LONG",
            entry_price=entry,
            stop_loss=stop,
            target1=target1,
            target2=target2,
            probability=0.65,
            risk_reward=2.0,
            strength=self._calculate_setup_strength(signal_idx, "LONG"),
            notes="Falling wedge with momentum divergence",
            context={"swing_lows": swing_lows, "momentum": momentum}
        )
        
    def find_spike_and_channel(self) -> List[Setup]:
        """Find spike and channel patterns"""
        setups = []
        df = self.data
        
        for i in range(10, len(df) - 30):
            # Bull spike and channel
            bull_pattern = self._find_bull_spike_channel(i)
            if bull_pattern:
                setups.append(bull_pattern)
                
            # Bear spike and channel
            bear_pattern = self._find_bear_spike_channel(i)
            if bear_pattern:
                setups.append(bear_pattern)
                
        return setups
        
    def _find_bull_spike_channel(self, idx: int) -> Optional[Setup]:
        """Find bull spike and channel pattern"""
        df = self.data
        
        # Check for spike (2-5 strong bull bars)
        spike_bars = 0
        spike_move = 0
        
        for i in range(idx, min(idx + 5, len(df))):
            if df['bull_bar'].iloc[i] and df['body'].iloc[i] > df['atr'].iloc[i] * 0.7:
                spike_bars += 1
                spike_move += df['close'].iloc[i] - df['open'].iloc[i]
            else:
                break
                
        if spike_bars < 2:
            return None
            
        # Look for channel after spike
        channel_start = idx + spike_bars
        if channel_start + 10 >= len(df):
            return None
            
        # Channel should be slower, grinding move
        channel_range = df['high'].iloc[channel_start:channel_start+10].max() - \
                       df['low'].iloc[channel_start:channel_start+10].min()
                       
        if channel_range > spike_move:
            return None  # Channel too wide
            
        # Look for test of channel bottom
        for i in range(channel_start + 10, min(channel_start + 30, len(df))):
            if df['low'].iloc[i] <= df['low'].iloc[channel_start:channel_start+3].min():
                # Double bottom setup found
                entry = df['high'].iloc[i] + 0.01
                stop = df['low'].iloc[i] - 0.01
                risk = entry - stop
                target1 = entry + risk
                target2 = entry + spike_move  # Measured move
                
                return Setup(
                    timestamp=df.index[i],
                    type=SetupType.SPIKE_AND_CHANNEL_BULL,
                    direction="LONG",
                    entry_price=entry,
                    stop_loss=stop,
                    target1=target1,
                    target2=target2,
                    probability=0.70,
                    risk_reward=target2/risk if risk > 0 else 2.0,
                    strength=self._calculate_setup_strength(i, "LONG"),
                    notes="Spike and channel bull, test of channel bottom",
                    context={
                        "spike_start": idx,
                        "spike_bars": spike_bars,
                        "channel_start": channel_start
                    }
                )
                
        return None
        
    def _find_bear_spike_channel(self, idx: int) -> Optional[Setup]:
        """Find bear spike and channel pattern"""
        df = self.data
        
        # Check for spike (2-5 strong bear bars)
        spike_bars = 0
        spike_move = 0
        
        for i in range(idx, min(idx + 5, len(df))):
            if df['bear_bar'].iloc[i] and df['body'].iloc[i] > df['atr'].iloc[i] * 0.7:
                spike_bars += 1
                spike_move += df['open'].iloc[i] - df['close'].iloc[i]
            else:
                break
                
        if spike_bars < 2:
            return None
            
        # Look for channel after spike
        channel_start = idx + spike_bars
        if channel_start + 10 >= len(df):
            return None
            
        # Channel should be slower, grinding move
        channel_range = df['high'].iloc[channel_start:channel_start+10].max() - \
                       df['low'].iloc[channel_start:channel_start+10].min()
                       
        if channel_range > spike_move:
            return None  # Channel too wide
            
        # Look for test of channel top
        for i in range(channel_start + 10, min(channel_start + 30, len(df))):
            if df['high'].iloc[i] >= df['high'].iloc[channel_start:channel_start+3].max():
                # Double top setup found
                entry = df['low'].iloc[i] - 0.01
                stop = df['high'].iloc[i] + 0.01
                risk = stop - entry
                target1 = entry - risk
                target2 = entry - spike_move  # Measured move
                
                return Setup(
                    timestamp=df.index[i],
                    type=SetupType.SPIKE_AND_CHANNEL_BEAR,
                    direction="SHORT",
                    entry_price=entry,
                    stop_loss=stop,
                    target1=target1,
                    target2=target2,
                    probability=0.70,
                    risk_reward=abs(target2-entry)/risk if risk > 0 else 2.0,
                    strength=self._calculate_setup_strength(i, "SHORT"),
                    notes="Spike and channel bear, test of channel top",
                    context={
                        "spike_start": idx,
                        "spike_bars": spike_bars,
                        "channel_start": channel_start
                    }
                )
                
        return None
        
    def find_failed_breakouts(self) -> List[Setup]:
        """Find failed breakout patterns"""
        setups = []
        df = self.data
        
        for i in range(20, len(df) - 5):
            # Failed bull breakout
            bull_fail = self._find_failed_bull_breakout(i)
            if bull_fail:
                setups.append(bull_fail)
                
            # Failed bear breakout
            bear_fail = self._find_failed_bear_breakout(i)
            if bear_fail:
                setups.append(bear_fail)
                
        return setups
        
    def _find_failed_bull_breakout(self, idx: int) -> Optional[Setup]:
        """Find failed bull breakout"""
        df = self.data
        lookback = 10
        
        # Find recent resistance
        recent_high = df['high'].iloc[idx-lookback:idx].max()
        
        # Check for breakout
        if df['high'].iloc[idx] <= recent_high:
            return None
            
        # Check for immediate failure (within 3 bars)
        for i in range(idx + 1, min(idx + 4, len(df))):
            if df['close'].iloc[i] < recent_high:
                # Failed breakout found
                entry = df['low'].iloc[i] - 0.01
                stop = df['high'].iloc[idx] + 0.01
                risk = stop - entry
                target1 = entry - risk
                target2 = entry - (risk * 2)
                
                return Setup(
                    timestamp=df.index[i],
                    type=SetupType.FAILED_BREAKOUT_BULL,
                    direction="SHORT",
                    entry_price=entry,
                    stop_loss=stop,
                    target1=target1,
                    target2=target2,
                    probability=0.75,
                    risk_reward=2.0,
                    strength=self._calculate_setup_strength(i, "SHORT"),
                    notes=f"Failed breakout above {recent_high:.2f}",
                    context={
                        "resistance": recent_high,
                        "breakout_bar": idx,
                        "failure_bar": i
                    }
                )
                
        return None
        
    def _find_failed_bear_breakout(self, idx: int) -> Optional[Setup]:
        """Find failed bear breakout"""
        df = self.data
        lookback = 10
        
        # Find recent support
        recent_low = df['low'].iloc[idx-lookback:idx].min()
        
        # Check for breakout
        if df['low'].iloc[idx] >= recent_low:
            return None
            
        # Check for immediate failure (within 3 bars)
        for i in range(idx + 1, min(idx + 4, len(df))):
            if df['close'].iloc[i] > recent_low:
                # Failed breakout found
                entry = df['high'].iloc[i] + 0.01
                stop = df['low'].iloc[idx] - 0.01
                risk = entry - stop
                target1 = entry + risk
                target2 = entry + (risk * 2)
                
                return Setup(
                    timestamp=df.index[i],
                    type=SetupType.FAILED_BREAKOUT_BEAR,
                    direction="LONG",
                    entry_price=entry,
                    stop_loss=stop,
                    target1=target1,
                    target2=target2,
                    probability=0.75,
                    risk_reward=2.0,
                    strength=self._calculate_setup_strength(i, "LONG"),
                    notes=f"Failed breakout below {recent_low:.2f}",
                    context={
                        "support": recent_low,
                        "breakout_bar": idx,
                        "failure_bar": i
                    }
                )
                
        return None
        
    def find_second_entries(self) -> List[Setup]:
        """Find second entry patterns (Brooks' favorite)"""
        setups = []
        df = self.data
        
        for i in range(30, len(df) - 2):
            # Bull second entry
            bull_second = self._find_bull_second_entry(i)
            if bull_second:
                setups.append(bull_second)
                
            # Bear second entry
            bear_second = self._find_bear_second_entry(i)
            if bear_second:
                setups.append(bear_second)
                
        return setups
        
    def _find_bull_second_entry(self, idx: int) -> Optional[Setup]:
        """Find bull second entry (failed Low 2)"""
        df = self.data
        
        # Need bull trend context
        if df['market_structure'].iloc[idx] != MarketStructure.BULL_TREND.value:
            return None
            
        # Look for Low 1 and Low 2 that failed
        low1_idx = None
        low2_idx = None
        
        # Find Low 1
        for i in range(idx - 15, idx - 3):
            if df['low'].iloc[i] < df['low'].iloc[i-1]:
                low1_idx = i
                break
                
        if not low1_idx:
            return None
            
        # Find Low 2 (should be higher low)
        for i in range(low1_idx + 2, idx):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and
                df['low'].iloc[i] > df['low'].iloc[low1_idx]):
                low2_idx = i
                break
                
        if not low2_idx:
            return None
            
        # Check if Low 2 failed (market went up instead)
        if df['high'].iloc[idx] > df['high'].iloc[low2_idx]:
            # Second entry setup
            entry = df['high'].iloc[idx] + 0.01
            stop = df['low'].iloc[low2_idx] - 0.01
            risk = entry - stop
            target1 = entry + risk
            target2 = entry + (risk * 2)
            
            return Setup(
                timestamp=df.index[idx],
                type=SetupType.SECOND_ENTRY_BULL,
                direction="LONG",
                entry_price=entry,
                stop_loss=stop,
                target1=target1,
                target2=target2,
                probability=0.70,
                risk_reward=2.0,
                strength=self._calculate_setup_strength(idx, "LONG"),
                notes="Second entry long (failed Low 2)",
                context={
                    "low1_idx": low1_idx,
                    "low2_idx": low2_idx
                }
            )
            
        return None
        
    def _find_bear_second_entry(self, idx: int) -> Optional[Setup]:
        """Find bear second entry (failed High 2)"""
        df = self.data
        
        # Need bear trend context
        if df['market_structure'].iloc[idx] != MarketStructure.BEAR_TREND.value:
            return None
            
        # Look for High 1 and High 2 that failed
        high1_idx = None
        high2_idx = None
        
        # Find High 1
        for i in range(idx - 15, idx - 3):
            if df['high'].iloc[i] > df['high'].iloc[i-1]:
                high1_idx = i
                break
                
        if not high1_idx:
            return None
            
        # Find High 2 (should be lower high)
        for i in range(high1_idx + 2, idx):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and
                df['high'].iloc[i] < df['high'].iloc[high1_idx]):
                high2_idx = i
                break
                
        if not high2_idx:
            return None
            
        # Check if High 2 failed (market went down instead)
        if df['low'].iloc[idx] < df['low'].iloc[high2_idx]:
            # Second entry setup
            entry = df['low'].iloc[idx] - 0.01
            stop = df['high'].iloc[high2_idx] + 0.01
            risk = stop - entry
            target1 = entry - risk
            target2 = entry - (risk * 2)
            
            return Setup(
                timestamp=df.index[idx],
                type=SetupType.SECOND_ENTRY_BEAR,
                direction="SHORT",
                entry_price=entry,
                stop_loss=stop,
                target1=target1,
                target2=target2,
                probability=0.70,
                risk_reward=2.0,
                strength=self._calculate_setup_strength(idx, "SHORT"),
                notes="Second entry short (failed High 2)",
                context={
                    "high1_idx": high1_idx,
                    "high2_idx": high2_idx
                }
            )
            
        return None
        
    def _calculate_setup_strength(self, idx: int, direction: str) -> float:
        """Calculate setup strength based on multiple factors"""
        df = self.data
        strength = 0.0
        
        # Market structure alignment
        if direction == "LONG":
            if df['market_structure'].iloc[idx] == MarketStructure.BULL_TREND.value:
                strength += 0.3
        else:
            if df['market_structure'].iloc[idx] == MarketStructure.BEAR_TREND.value:
                strength += 0.3
                
        # Signal bar quality
        bar_analysis = self.bar_analyses[idx-1] if idx > 0 else None
        if bar_analysis:
            strength += bar_analysis.strength * 0.2
            
        # Distance from EMA
        ema_distance = abs(df['close'].iloc[idx] - df['ema'].iloc[idx]) / df['atr'].iloc[idx]
        if ema_distance < 1:  # Close to EMA is good for with-trend
            strength += 0.2
        elif ema_distance > 3:  # Far from EMA might mean exhaustion
            strength -= 0.1
            
        # Recent momentum
        recent_bars = 5
        if direction == "LONG":
            bull_bars = df['bull_bar'].iloc[idx-recent_bars:idx].sum()
            strength += (bull_bars / recent_bars) * 0.2
        else:
            bear_bars = df['bear_bar'].iloc[idx-recent_bars:idx].sum()
            strength += (bear_bars / recent_bars) * 0.2
            
        # Volume confirmation (if available)
        if 'volume' in df.columns:
            vol_ratio = df['volume'].iloc[idx] / df['volume'].iloc[idx-20:idx].mean()
            if vol_ratio > 1.5:
                strength += 0.1
                
        return min(1.0, max(0.0, strength))
        
    def find_all_setups(self) -> List[Setup]:
        """Find all Brooks setups"""
        all_setups = []
        
        # High/Low patterns
        all_setups.extend(self.find_high_low_patterns())
        
        # Wedges
        all_setups.extend(self.find_wedge_patterns())
        
        # Spike and Channel
        all_setups.extend(self.find_spike_and_channel())
        
        # Failed Breakouts
        all_setups.extend(self.find_failed_breakouts())
        
        # Second Entries
        all_setups.extend(self.find_second_entries())
        
        # Sort by timestamp
        all_setups.sort(key=lambda x: x.timestamp)
        
        return all_setups
        
    def filter_setups(self, 
                     setups: List[Setup],
                     min_probability: float = 0.6,
                     min_risk_reward: float = 1.5,
                     min_strength: float = 0.5) -> List[Setup]:
        """Filter setups based on quality criteria"""
        filtered = []
        
        for setup in setups:
            if (setup.probability >= min_probability and
                setup.risk_reward >= min_risk_reward and
                setup.strength >= min_strength):
                filtered.append(setup)
                
        return filtered
        
    def generate_signals(self, 
                        min_probability: float = 0.6,
                        min_risk_reward: float = 1.5,
                        min_strength: float = 0.5) -> pd.DataFrame:
        """Generate trading signals DataFrame"""
        # Find all setups
        all_setups = self.find_all_setups()
        
        # Filter by quality
        filtered = self.filter_setups(
            all_setups, 
            min_probability, 
            min_risk_reward, 
            min_strength
        )
        
        # Convert to DataFrame
        if not filtered:
            return pd.DataFrame()
            
        signals_data = []
        for setup in filtered:
            signals_data.append({
                'timestamp': setup.timestamp,
                'type': setup.type.value,
                'direction': setup.direction,
                'entry': setup.entry_price,
                'stop': setup.stop_loss,
                'target1': setup.target1,
                'target2': setup.target2,
                'probability': setup.probability,
                'risk_reward': setup.risk_reward,
                'strength': setup.strength,
                'notes': setup.notes
            })
            
        return pd.DataFrame(signals_data)
        
    def backtest(self, 
                signals: pd.DataFrame,
                initial_capital: float = 10000,
                risk_per_trade: float = 0.02,
                commission: float = 0.001) -> Dict:
        """
        Backtest signals with Brooks' money management
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Trading signals to backtest
        initial_capital : float
            Starting capital
        risk_per_trade : float
            Risk per trade (Brooks recommends 2%)
        commission : float
            Commission per trade
        """
        if signals.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_capital': initial_capital,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades': []
            }
            
        trades = []
        capital = initial_capital
        peak_capital = capital
        max_drawdown = 0
        
        for _, signal in signals.iterrows():
            # Calculate position size (Brooks' 2% rule)
            risk_amount = capital * risk_per_trade
            risk_per_share = abs(signal['entry'] - signal['stop'])
            
            if risk_per_share <= 0:
                continue
                
            shares = int(risk_amount / risk_per_share)
            if shares <= 0:
                continue
                
            # Simulate trade execution
            trade_result = self._simulate_trade(signal, shares)
            
            # Update capital
            capital += trade_result['pnl']
            capital -= abs(trade_result['pnl']) * commission
            
            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
            
            # Record trade
            trade_result['capital_after'] = capital
            trades.append(trade_result)
            
        # Calculate statistics
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        if not trades_df.empty:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
            
            # Sharpe ratio
            returns = trades_df['pnl'] / initial_capital
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            sharpe = 0
            
        return {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': capital - initial_capital,
            'final_capital': capital,
            'total_return': (capital / initial_capital - 1) * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'trades': trades_df
        }
        
    def _simulate_trade(self, signal: pd.Series, shares: int) -> Dict:
        """Simulate individual trade execution"""
        # Find entry bar
        entry_idx = self.data.index.get_loc(signal['timestamp'])
        
        # Simulate forward bars
        entry_filled = False
        exit_price = None
        exit_reason = None
        bars_held = 0
        
        for i in range(entry_idx + 1, min(entry_idx + 50, len(self.data))):
            bar = self.data.iloc[i]
            bars_held += 1
            
            # Check entry
            if not entry_filled:
                if signal['direction'] == 'LONG':
                    if bar['high'] >= signal['entry']:
                        entry_filled = True
                else:  # SHORT
                    if bar['low'] <= signal['entry']:
                        entry_filled = True
                continue
                
            # Check exit conditions
            if signal['direction'] == 'LONG':
                # Stop loss
                if bar['low'] <= signal['stop']:
                    exit_price = signal['stop']
                    exit_reason = 'STOP_LOSS'
                    break
                # Target 1
                elif bar['high'] >= signal['target1'] and bars_held < 10:
                    exit_price = signal['target1']
                    exit_reason = 'TARGET_1'
                    break
                # Target 2
                elif bar['high'] >= signal['target2']:
                    exit_price = signal['target2']
                    exit_reason = 'TARGET_2'
                    break
                    
            else:  # SHORT
                # Stop loss
                if bar['high'] >= signal['stop']:
                    exit_price = signal['stop']
                    exit_reason = 'STOP_LOSS'
                    break
                # Target 1
                elif bar['low'] <= signal['target1'] and bars_held < 10:
                    exit_price = signal['target1']
                    exit_reason = 'TARGET_1'
                    break
                # Target 2
                elif bar['low'] <= signal['target2']:
                    exit_price = signal['target2']
                    exit_reason = 'TARGET_2'
                    break
                    
        # Calculate P&L
        if entry_filled and exit_price:
            if signal['direction'] == 'LONG':
                pnl = (exit_price - signal['entry']) * shares
            else:
                pnl = (signal['entry'] - exit_price) * shares
        else:
            pnl = 0
            exit_reason = 'NO_EXIT'
            
        return {
            'timestamp': signal['timestamp'],
            'type': signal['type'],
            'direction': signal['direction'],
            'entry': signal['entry'],
            'exit': exit_price,
            'shares': shares,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'bars_held': bars_held
        }


# Testing function
def test_brooks_system():
    """Test the complete Brooks system"""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    
    # Create trending market with realistic price action
    trend = np.cumsum(np.random.randn(1000) * 0.3 + 0.01)
    cycles = np.sin(np.linspace(0, 20*np.pi, 1000)) * 2
    noise = np.random.randn(1000) * 0.5
    
    price = 100 + trend + cycles + noise
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = price
    data['open'] = price + np.random.randn(1000) * 0.2
    data['high'] = np.maximum(data['open'], data['close']) + abs(np.random.randn(1000) * 0.3)
    data['low'] = np.minimum(data['open'], data['close']) - abs(np.random.randn(1000) * 0.3)
    data['volume'] = np.random.randint(10000, 100000, 1000)
    
    # Initialize system
    brooks = AlBrooksPriceAction(data)
    
    # Generate signals
    signals = brooks.generate_signals(
        min_probability=0.6,
        min_risk_reward=1.5,
        min_strength=0.5
    )
    
    print("