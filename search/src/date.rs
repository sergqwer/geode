// SPDX-License-Identifier: AGPL-3.0-or-later
//! The dates engines give (ISO 8601 from the APIs, "5 Mar 2024" from Startpage), as Python's datetime holds them,
//! written the way `datetime.isoformat()` writes them (the publishedDate of the output).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DateTime {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub micro: u32,
    /// offset from UTC in minutes; None for a naive time
    pub offset: Option<i32>,
}

impl DateTime {
    pub fn isoformat(&self) -> String {
        let mut s = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        );
        if self.micro != 0 {
            s.push_str(&format!(".{:06}", self.micro));
        }
        if let Some(o) = self.offset {
            let sign = if o < 0 { '-' } else { '+' };
            s.push_str(&format!("{sign}{:02}:{:02}", o.abs() / 60, o.abs() % 60));
        }
        s
    }

    fn valid(&self) -> bool {
        (1..=12).contains(&self.month)
            && self.day >= 1
            && self.day <= days_in_month(self.year, self.month)
            && self.hour < 24
            && self.minute < 60
            && self.second < 60
    }
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        _ => 28,
    }
}

fn num(s: &str) -> Option<u32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// ISO 8601 as datetime.fromisoformat (3.11+) and dateutil read it: date, optional time (T or space), optional
/// fraction, optional Z / ±HH:MM / ±HHMM.
pub fn parse_iso(s: &str) -> Option<DateTime> {
    let s = s.trim();
    if s.len() < 10 {
        return None;
    }
    let (date, rest) = s.split_at(10);
    let mut d = DateTime {
        year: num(&date[0..4])? as i32,
        month: num(&date[5..7])?,
        day: num(&date[8..10])?,
        hour: 0,
        minute: 0,
        second: 0,
        micro: 0,
        offset: None,
    };
    if &date[4..5] != "-" || &date[7..8] != "-" {
        return None;
    }
    let rest = rest.strip_prefix(['T', ' ']).unwrap_or(rest);
    if !rest.is_empty() {
        // the zone: Z or the last + / - after the time
        let (time, zone) = if let Some(t) = rest.strip_suffix('Z').or_else(|| rest.strip_suffix('z')) {
            (t, Some(0))
        } else if let Some(i) = rest.rfind(['+', '-']) {
            let z = &rest[i + 1..];
            let (h, m) = if let Some((h, m)) = z.split_once(':') {
                (num(h)?, num(m)?)
            } else if z.len() == 4 {
                (num(&z[..2])?, num(&z[2..])?)
            } else {
                (num(z)?, 0)
            };
            let o = (h * 60 + m) as i32;
            (&rest[..i], Some(if rest.as_bytes()[i] == b'-' { -o } else { o }))
        } else {
            (rest, None)
        };
        d.offset = zone;
        if !time.is_empty() {
            let (hms, frac) = time.split_once(['.', ',']).unwrap_or((time, ""));
            let p: Vec<&str> = hms.split(':').collect();
            d.hour = num(p.first()?)?;
            d.minute = p.get(1).map(|x| num(x)).unwrap_or(Some(0))?;
            d.second = p.get(2).map(|x| num(x)).unwrap_or(Some(0))?;
            if !frac.is_empty() {
                let f: String = frac.chars().chain(std::iter::repeat('0')).take(6).collect();
                d.micro = num(&f)?;
            }
        }
    }
    d.valid().then_some(d)
}

const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

/// dateutil.parser.parse(dayfirst=True) for "5 Mar 2024" (Startpage's dates)
pub fn parse_day_month_year(s: &str) -> Option<DateTime> {
    let p: Vec<&str> = s.split_whitespace().collect();
    if p.len() != 3 {
        return None;
    }
    let month = MONTHS.iter().position(|m| p[1].to_lowercase().starts_with(m))? as u32 + 1;
    let d = DateTime {
        year: num(p[2])? as i32,
        month,
        day: num(p[0])?,
        hour: 0,
        minute: 0,
        second: 0,
        micro: 0,
        offset: None,
    };
    d.valid().then_some(d)
}

/// Now, in UTC, as a naive datetime (datetime.now() on a phone whose clock is UTC).
pub fn now() -> DateTime {
    let secs =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    from_unix(secs)
}

pub fn from_unix(secs: i64) -> DateTime {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // civil from days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    DateTime {
        year: (if m <= 2 { y + 1 } else { y }) as i32,
        month: m,
        day: d,
        hour: (rem / 3600) as u32,
        minute: (rem % 3600 / 60) as u32,
        second: (rem % 60) as u32,
        micro: 0,
        offset: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_as_python_writes_it() {
        assert_eq!(parse_iso("2024-12-06T15:37:31.123456Z").unwrap().isoformat(), "2024-12-06T15:37:31.123456+00:00");
        assert_eq!(parse_iso("2017-07-23T20:00:27").unwrap().isoformat(), "2017-07-23T20:00:27");
        assert_eq!(parse_iso("2023-01-02T03:04:05+05:30").unwrap().isoformat(), "2023-01-02T03:04:05+05:30");
        assert_eq!(parse_iso("2023-01-02T03:04:05-0800").unwrap().isoformat(), "2023-01-02T03:04:05-08:00");
        assert_eq!(parse_iso("2023-01-02").unwrap().isoformat(), "2023-01-02T00:00:00");
        assert_eq!(parse_iso("2023-01-02T03:04:05.5").unwrap().micro, 500000);
        assert!(parse_iso("2023-02-30T00:00:00").is_none());
        assert!(parse_iso("yesterday").is_none());
    }

    #[test]
    fn other_forms() {
        assert_eq!(parse_day_month_year("5 Mar 2024").unwrap().isoformat(), "2024-03-05T00:00:00");
        assert_eq!(from_unix(0).isoformat(), "1970-01-01T00:00:00");
        assert_eq!(from_unix(1709596800).isoformat(), "2024-03-05T00:00:00");
    }
}
