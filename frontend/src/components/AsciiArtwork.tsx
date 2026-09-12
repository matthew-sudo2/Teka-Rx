import type { SVGProps } from "react";

type ArtworkProps = Omit<SVGProps<SVGSVGElement>, "children"> & {
  /**
   * Supply a short title only when the artwork communicates information.
   * Without a title, the SVG is treated as decorative and hidden from AT.
   */
  accessibleTitle?: string;
};

type DotTone = "green" | "white" | "muted" | "danger";

type Dot = {
  x: number;
  y: number;
  radius: number;
  opacity: number;
  tone: DotTone;
};

const round = (value: number) => Number(value.toFixed(2));

function pointsAlongLine(
  start: readonly [number, number],
  end: readonly [number, number],
  spacing: number,
  tone: DotTone,
  radius = 1.65,
  opacity = 0.92,
): Dot[] {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy) / spacing));

  return Array.from({ length: steps + 1 }, (_, index) => {
    const progress = index / steps;
    return {
      x: round(start[0] + dx * progress),
      y: round(start[1] + dy * progress),
      radius,
      opacity,
      tone,
    };
  });
}

function pointsAlongPolyline(
  points: ReadonlyArray<readonly [number, number]>,
  spacing: number,
  tone: DotTone,
  radius = 1.65,
  opacity = 0.92,
): Dot[] {
  return points.flatMap((point, index) => {
    const next = points[index + 1];
    return next
      ? pointsAlongLine(point, next, spacing, tone, radius, opacity).slice(
          index === 0 ? 0 : 1,
        )
      : [];
  });
}

function mergeClassNames(...names: Array<string | undefined>) {
  return names.filter(Boolean).join(" ");
}

function artworkAccessibility(accessibleTitle?: string) {
  return accessibleTitle
    ? ({ role: "img", "aria-label": accessibleTitle } as const)
    : ({ "aria-hidden": true } as const);
}

type Vector3 = {
  x: number;
  y: number;
  z: number;
};

type ProjectedAsciiGlyph = {
  x: number;
  y: number;
  z: number;
  glyph: string;
  tone: "green" | "white";
  opacity: number;
  fontSize: number;
  fontWeight: 500 | 600 | 700;
  seam: boolean;
};

const ASCII_PILL_CENTER_X = 430;
const ASCII_PILL_CENTER_Y = 270;
const ASCII_PILL_BODY_HALF_LENGTH = 190;
const ASCII_PILL_RADIUS = 105;
const ASCII_PILL_CAMERA_DISTANCE = 3200;
const ASCII_PILL_TILT_Y = (18 * Math.PI) / 180;
const ASCII_PILL_TILT_X = (-5 * Math.PI) / 180;
const ASCII_PILL_SCREEN_ROTATION = (-31 * Math.PI) / 180;

function stableCellHash(column: number, row: number, salt = 0) {
  const columnHash = Math.imul(column + 41, 73_856_093);
  const rowHash = Math.imul(row + 59, 19_349_663);
  const saltHash = Math.imul(salt + 17, 83_492_791);
  return (columnHash ^ rowHash ^ saltHash) >>> 0;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizeVector(vector: Vector3): Vector3 {
  const length = Math.hypot(vector.x, vector.y, vector.z) || 1;
  return {
    x: vector.x / length,
    y: vector.y / length,
    z: vector.z / length,
  };
}

function rotateCapsuleVector(vector: Vector3): Vector3 {
  const cosY = Math.cos(ASCII_PILL_TILT_Y);
  const sinY = Math.sin(ASCII_PILL_TILT_Y);
  const tiltedY = {
    x: vector.x * cosY + vector.z * sinY,
    y: vector.y,
    z: -vector.x * sinY + vector.z * cosY,
  };

  const cosX = Math.cos(ASCII_PILL_TILT_X);
  const sinX = Math.sin(ASCII_PILL_TILT_X);
  const tiltedX = {
    x: tiltedY.x,
    y: tiltedY.y * cosX - tiltedY.z * sinX,
    z: tiltedY.y * sinX + tiltedY.z * cosX,
  };

  const cosZ = Math.cos(ASCII_PILL_SCREEN_ROTATION);
  const sinZ = Math.sin(ASCII_PILL_SCREEN_ROTATION);
  return {
    x: tiltedX.x * cosZ - tiltedX.y * sinZ,
    y: tiltedX.x * sinZ + tiltedX.y * cosZ,
    z: tiltedX.z,
  };
}

const ASCII_PILL_LIGHT = normalizeVector({ x: -0.42, y: -0.5, z: 0.76 });

function chooseAsciiGlyph(
  light: number,
  rim: number,
  frontness: number,
  hash: number,
  seam: boolean,
) {
  if (seam) {
    return frontness > 0.48 ? ["@", "#", "%"][hash % 3] : ["*", "+", "="][hash % 3];
  }

  const score = clamp(light * 0.58 + rim * 0.3 + frontness * 0.12, 0, 1);
  if (score > 0.82) return ["@", "#", "%"][hash % 3];
  if (score > 0.65) return ["#", "*", "%"][hash % 3];
  if (score > 0.48) return ["*", "+", "="][hash % 3];
  if (score > 0.3) return ["+", ":", "-"][hash % 3];
  return [":", "-", "."][hash % 3];
}

function projectAsciiGlyph(
  point: Vector3,
  normal: Vector3,
  tone: "green" | "white",
  sampleA: number,
  sampleB: number,
  seam = false,
): ProjectedAsciiGlyph | null {
  const rotatedPoint = rotateCapsuleVector(point);
  const rotatedNormal = normalizeVector(rotateCapsuleVector(normal));
  const frontness = clamp((rotatedNormal.z + 1) / 2, 0, 1);
  const rim = Math.exp(-Math.abs(rotatedNormal.z) * 4.6);
  const hash = stableCellHash(sampleA, sampleB, seam ? 91 : 23);

  // Preserve a restrained trace of the rear surface so the ASCII channels
  // wrap around the volume, while dropping enough glyphs to avoid a flat fill.
  if (!seam && rotatedNormal.z < -0.2 && rim < 0.28 && hash % 3 !== 0) {
    return null;
  }

  const perspective = ASCII_PILL_CAMERA_DISTANCE /
    (ASCII_PILL_CAMERA_DISTANCE - rotatedPoint.z);
  const light = clamp(
    rotatedNormal.x * ASCII_PILL_LIGHT.x +
      rotatedNormal.y * ASCII_PILL_LIGHT.y +
      rotatedNormal.z * ASCII_PILL_LIGHT.z,
    0,
    1,
  );
  const depth = clamp((rotatedPoint.z + 260) / 520, 0, 1);
  const opacity = seam
    ? clamp(0.3 + frontness * 0.5 + rim * 0.2, 0.3, 1)
    : clamp(0.18 + frontness * 0.39 + light * 0.25 + rim * 0.18, 0.18, 0.96);
  const fontSize = seam
    ? 13.7 + depth * 1.6
    : 11.2 + depth * 2.2 + rim * 1.1;

  return {
    x: round(ASCII_PILL_CENTER_X + rotatedPoint.x * perspective),
    y: round(ASCII_PILL_CENTER_Y + rotatedPoint.y * perspective),
    z: rotatedPoint.z,
    glyph: chooseAsciiGlyph(light, rim, frontness, hash, seam),
    tone,
    opacity: round(opacity),
    fontSize: round(fontSize),
    fontWeight: seam || light + rim > 1.12 ? 700 : light > 0.42 ? 600 : 500,
    seam,
  };
}

function buildAsciiPillGlyphs() {
  const glyphs: ProjectedAsciiGlyph[] = [];
  const cylinderSlices = 27;
  const cylinderAngles = 28;
  const capRings = 11;

  for (let slice = 0; slice < cylinderSlices; slice += 1) {
    const progress = slice / (cylinderSlices - 1);
    const x = -ASCII_PILL_BODY_HALF_LENGTH + progress * ASCII_PILL_BODY_HALF_LENGTH * 2;
    const twist = (progress - 0.5) * 0.82;

    for (let angleIndex = 0; angleIndex < cylinderAngles; angleIndex += 1) {
      const theta = (angleIndex / cylinderAngles) * Math.PI * 2 + twist;
      const point = {
        x,
        y: ASCII_PILL_RADIUS * Math.cos(theta),
        z: ASCII_PILL_RADIUS * Math.sin(theta),
      };
      const normal = { x: 0, y: Math.cos(theta), z: Math.sin(theta) };
      const glyph = projectAsciiGlyph(
        point,
        normal,
        x <= 0 ? "white" : "green",
        slice,
        angleIndex,
      );
      if (glyph) glyphs.push(glyph);
    }
  }

  for (const direction of [-1, 1] as const) {
    for (let ring = 1; ring <= capRings; ring += 1) {
      const alpha = (ring / (capRings + 1)) * (Math.PI / 2);
      const radial = ASCII_PILL_RADIUS * Math.cos(alpha);
      const angleCount = Math.max(9, Math.round(cylinderAngles * Math.cos(alpha)));

      for (let angleIndex = 0; angleIndex < angleCount; angleIndex += 1) {
        const theta =
          (angleIndex / angleCount) * Math.PI * 2 + direction * alpha * 0.34;
        const point = {
          x:
            direction *
            (ASCII_PILL_BODY_HALF_LENGTH + ASCII_PILL_RADIUS * Math.sin(alpha)),
          y: radial * Math.cos(theta),
          z: radial * Math.sin(theta),
        };
        const normal = {
          x: direction * Math.sin(alpha),
          y: Math.cos(alpha) * Math.cos(theta),
          z: Math.cos(alpha) * Math.sin(theta),
        };
        const glyph = projectAsciiGlyph(
          point,
          normal,
          direction < 0 ? "white" : "green",
          ring + (direction > 0 ? 37 : 0),
          angleIndex,
        );
        if (glyph) glyphs.push(glyph);
      }
    }

    const tipGlyph = projectAsciiGlyph(
      {
        x: direction * (ASCII_PILL_BODY_HALF_LENGTH + ASCII_PILL_RADIUS),
        y: 0,
        z: 0,
      },
      { x: direction, y: 0, z: 0 },
      direction < 0 ? "white" : "green",
      direction < 0 ? 83 : 89,
      0,
    );
    if (tipGlyph) glyphs.push(tipGlyph);
  }

  // Two close rings make the join feel structural. Perspective turns these
  // circular 3D rings into the bright elliptical seam seen in the reference.
  for (const seamOffset of [-3.2, 3.2]) {
    const seamAngles = 48;
    for (let angleIndex = 0; angleIndex < seamAngles; angleIndex += 1) {
      const theta = (angleIndex / seamAngles) * Math.PI * 2;
      const glyph = projectAsciiGlyph(
        {
          x: seamOffset,
          y: ASCII_PILL_RADIUS * Math.cos(theta),
          z: ASCII_PILL_RADIUS * Math.sin(theta),
        },
        { x: 0, y: Math.cos(theta), z: Math.sin(theta) },
        "white",
        seamOffset < 0 ? 97 : 101,
        angleIndex,
        true,
      );
      if (glyph) glyphs.push(glyph);
    }
  }

  return glyphs.sort((left, right) => left.z - right.z);
}

const asciiPillGlyphs = buildAsciiPillGlyphs();

const chartDots = (() => {
  const dots: Dot[] = [];
  const baseline = 150;
  const barData = [
    { x: 23, height: 40, tone: "white" as const },
    { x: 55, height: 63, tone: "white" as const },
    { x: 87, height: 54, tone: "green" as const },
    { x: 119, height: 92, tone: "white" as const },
    { x: 151, height: 119, tone: "green" as const },
  ];

  for (const bar of barData) {
    for (let y = baseline; y >= baseline - bar.height; y -= 8) {
      dots.push({
        x: bar.x,
        y,
        radius: 1.7,
        opacity: y === baseline ? 0.58 : 0.9,
        tone: bar.tone,
      });
      dots.push({
        x: bar.x + 8,
        y,
        radius: 1.35,
        opacity: 0.72,
        tone: bar.tone,
      });
    }
  }

  dots.push(
    ...pointsAlongPolyline(
      [
        [18, 114],
        [48, 99],
        [78, 66],
        [108, 80],
        [140, 45],
        [160, 22],
      ],
      7,
      "green",
      1.7,
      0.95,
    ),
  );

  return dots;
})();

const riskDots = [
  ...pointsAlongPolyline(
    [
      [48, 8],
      [88, 80],
      [8, 80],
      [48, 8],
    ],
    6,
    "danger",
    1.75,
  ),
  ...pointsAlongLine([48, 31], [48, 57], 5, "danger", 1.8),
  { x: 48, y: 68, radius: 2.9, opacity: 1, tone: "danger" as const },
];

const brainDots = [
  ...pointsAlongPolyline(
    [
      [47, 14],
      [35, 8],
      [22, 14],
      [18, 25],
      [9, 31],
      [8, 45],
      [14, 53],
      [12, 65],
      [22, 76],
      [35, 78],
      [47, 70],
    ],
    5.4,
    "green",
    1.65,
  ),
  ...pointsAlongPolyline(
    [
      [49, 14],
      [61, 8],
      [74, 14],
      [78, 25],
      [87, 31],
      [88, 45],
      [82, 53],
      [84, 65],
      [74, 76],
      [61, 78],
      [49, 70],
    ],
    5.4,
    "green",
    1.65,
  ),
  ...pointsAlongLine([48, 15], [48, 70], 6, "green", 1.45, 0.82),
  ...pointsAlongPolyline(
    [
      [19, 27],
      [31, 31],
      [36, 43],
      [27, 51],
      [17, 49],
    ],
    5.2,
    "green",
    1.45,
    0.84,
  ),
  ...pointsAlongPolyline(
    [
      [77, 27],
      [65, 31],
      [60, 43],
      [69, 51],
      [79, 49],
    ],
    5.2,
    "green",
    1.45,
    0.84,
  ),
  ...pointsAlongPolyline(
    [
      [24, 68],
      [34, 61],
      [39, 51],
    ],
    5.2,
    "green",
    1.45,
    0.76,
  ),
  ...pointsAlongPolyline(
    [
      [72, 68],
      [62, 61],
      [57, 51],
    ],
    5.2,
    "green",
    1.45,
    0.76,
  ),
];

const evidenceDots = [
  ...pointsAlongPolyline(
    [
      [18, 8],
      [61, 8],
      [80, 27],
      [80, 82],
      [18, 82],
      [18, 8],
    ],
    5.5,
    "green",
    1.6,
  ),
  ...pointsAlongPolyline(
    [
      [61, 8],
      [61, 27],
      [80, 27],
    ],
    5,
    "green",
    1.5,
  ),
  ...pointsAlongLine([31, 42], [66, 42], 6, "green", 1.45, 0.88),
  ...pointsAlongLine([31, 54], [66, 54], 6, "green", 1.45, 0.88),
  ...pointsAlongLine([31, 66], [58, 66], 6, "green", 1.45, 0.88),
];

const toneClass = (tone: DotTone) => `ascii-dot ascii-dot--${tone}`;

export function AsciiPill({
  accessibleTitle,
  className,
  ...svgProps
}: ArtworkProps) {
  return (
    <svg
      {...svgProps}
      {...artworkAccessibility(accessibleTitle)}
      className={mergeClassNames("ascii-artwork", "ascii-pill", className)}
      focusable="false"
      viewBox="0 0 860 540"
      preserveAspectRatio="xMidYMid meet"
    >
      {accessibleTitle ? <title>{accessibleTitle}</title> : null}
      <g
        className="ascii-pill__glyph-field"
        fontFamily="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace"
      >
        {asciiPillGlyphs.map((glyph, index) => (
          <text
            className={mergeClassNames(
              `ascii-dot--${glyph.tone}`,
              "ascii-pill__glyph",
              glyph.seam ? "ascii-pill__glyph--seam" : undefined,
            )}
            dominantBaseline="central"
            fill="currentColor"
            fontSize={glyph.fontSize}
            fontWeight={glyph.fontWeight}
            key={`pill-glyph-${index}`}
            opacity={glyph.opacity}
            textAnchor="middle"
            x={glyph.x}
            y={glyph.y}
          >
            {glyph.glyph}
          </text>
        ))}
      </g>
    </svg>
  );
}

export function AsciiMiniChart({
  accessibleTitle,
  className,
  ...svgProps
}: ArtworkProps) {
  return (
    <svg
      {...svgProps}
      {...artworkAccessibility(accessibleTitle)}
      className={mergeClassNames("ascii-artwork", "ascii-mini-chart", className)}
      focusable="false"
      viewBox="0 0 180 170"
      preserveAspectRatio="xMidYMid meet"
    >
      {accessibleTitle ? <title>{accessibleTitle}</title> : null}
      {chartDots.map((dot, index) => (
        <circle
          className={toneClass(dot.tone)}
          cx={dot.x}
          cy={dot.y}
          fill="currentColor"
          key={`chart-${index}`}
          opacity={dot.opacity}
          r={dot.radius}
        />
      ))}
    </svg>
  );
}

export type DottedFeatureIconKind = "risk" | "brain" | "evidence";

export type DottedFeatureIconProps = ArtworkProps & {
  kind: DottedFeatureIconKind;
};

const iconDots: Record<DottedFeatureIconKind, Dot[]> = {
  risk: riskDots,
  brain: brainDots,
  evidence: evidenceDots,
};

export function DottedFeatureIcon({
  accessibleTitle,
  className,
  kind,
  ...svgProps
}: DottedFeatureIconProps) {
  return (
    <svg
      {...svgProps}
      {...artworkAccessibility(accessibleTitle)}
      className={mergeClassNames(
        "ascii-artwork",
        "dotted-feature-icon",
        `dotted-feature-icon--${kind}`,
        className,
      )}
      focusable="false"
      viewBox="0 0 96 90"
      preserveAspectRatio="xMidYMid meet"
    >
      {accessibleTitle ? <title>{accessibleTitle}</title> : null}
      {iconDots[kind].map((dot, index) => (
        <circle
          className={toneClass(dot.tone)}
          cx={dot.x}
          cy={dot.y}
          fill="currentColor"
          key={`${kind}-${index}`}
          opacity={dot.opacity}
          r={dot.radius}
        />
      ))}
    </svg>
  );
}

export function AsciiWarning(props: ArtworkProps) {
  return <DottedFeatureIcon {...props} kind="risk" />;
}

export function AsciiBrain(props: ArtworkProps) {
  return <DottedFeatureIcon {...props} kind="brain" />;
}

export function AsciiDocument(props: ArtworkProps) {
  return <DottedFeatureIcon {...props} kind="evidence" />;
}
