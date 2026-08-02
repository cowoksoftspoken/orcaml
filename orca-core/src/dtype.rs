use std::fmt;

/// The data type of a tensor's elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit floating point (Brain Float)
    BF16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

impl DType {
    /// Promotes this type with another type, returning the type that can represent both without loss of precision.
    /// Following general ML framework type promotion rules.
    pub fn promote(self, other: Self) -> Self {
        if self == other {
            return self;
        }

        match (self, other) {
            (DType::F64, _) | (_, DType::F64) => DType::F64,
            (DType::F32, _) | (_, DType::F32) => DType::F32,
            (DType::BF16, _) | (_, DType::BF16) => DType::BF16,
            (DType::F16, _) | (_, DType::F16) => DType::F16,
            (DType::I64, _) | (_, DType::I64) => DType::I64,
            (DType::I32, _) | (_, DType::I32) => DType::I32,
            (DType::U8, _) | (_, DType::U8) => DType::U8,
            _ => DType::F32, // Fallback
        }
    }

    /// Get the size of this type in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F16 | Self::BF16 => 2,
            Self::U8 | Self::Bool => 1,
        }
    }

    /// Convert a 32-bit float to a 16-bit float (IEEE 754).
    pub fn f32_to_f16(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let raw_exponent = (bits >> 23) & 0xff;
        let mantissa = bits & 0x7fffff;

        if raw_exponent == 0xff {
            let payload = mantissa >> 13;
            return (sign | 0x7c00 | payload | u32::from(payload == 0)) as u16;
        }

        let exponent = raw_exponent as i32 - 127 + 15;
        if exponent <= 0 {
            if exponent < -10 {
                return sign as u16;
            }

            let shift = (14 - exponent) as u32;
            let extended = mantissa | 0x800000;
            let mut half_mantissa = extended >> shift;
            let remainder = extended & ((1 << shift) - 1);
            let halfway = 1 << (shift - 1);
            if remainder > halfway || (remainder == halfway && half_mantissa & 1 != 0) {
                half_mantissa += 1;
            }
            return (sign | half_mantissa) as u16;
        }

        if exponent >= 0x1f {
            return (sign | 0x7c00) as u16;
        }

        let mut half_mantissa = mantissa >> 13;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && half_mantissa & 1 != 0) {
            half_mantissa += 1;
            if half_mantissa == 0x400 {
                return (sign | ((exponent as u32 + 1) << 10)) as u16;
            }
        }

        (sign | ((exponent as u32) << 10) | half_mantissa) as u16
    }

    /// Convert a 16-bit float (IEEE 754) to a 32-bit float
    pub fn f16_to_f32(val: u16) -> f32 {
        let sign = (val as u32 & 0x8000) << 16;
        let exponent = (val as u32 & 0x7c00) >> 10;
        let mantissa = val as u32 & 0x03ff;

        if exponent == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign);
            }
            // Subnormal
            let mut m = mantissa;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let new_exp = 127 - 15 - e + 1;
            let new_mant = (m & 0x3ff) << 13;
            return f32::from_bits(sign | (new_exp << 23) | new_mant);
        } else if exponent == 31 {
            return f32::from_bits(sign | 0x7f800000 | (mantissa << 13));
        }

        let new_exp = exponent + 127 - 15;
        let new_mant = mantissa << 13;
        f32::from_bits(sign | (new_exp << 23) | new_mant)
    }

    /// Convert a 32-bit float to a bfloat16 using round-to-nearest-even.
    pub fn f32_to_bf16(val: f32) -> u16 {
        let bits = val.to_bits();
        let rounding_bias = 0x7fff + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding_bias);
        let mut result = (rounded >> 16) as u16;
        if (result & 0x7f80) == 0x7f80 && (result & 0x007f) == 0 {
            result |= 1;
        }
        result
    }

    /// Convert a bfloat16 to a 32-bit float
    pub fn bf16_to_f32(val: u16) -> f32 {
        f32::from_bits((val as u32) << 16)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::Bool => "bool",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn f16_conversion_roundtrips_common_values() {
        for value in [0.0, -0.0, 1.0, -2.5, 65504.0] {
            let converted = DType::f16_to_f32(DType::f32_to_f16(value));
            assert_eq!(converted, value);
        }
    }

    #[test]
    fn bf16_conversion_rounds_and_preserves_nan() {
        let rounded = DType::bf16_to_f32(DType::f32_to_bf16(1.003_906_3));
        assert_eq!(rounded, 1.0);
        assert!(DType::bf16_to_f32(DType::f32_to_bf16(f32::NAN)).is_nan());
    }
}
