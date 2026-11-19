"use client";

// 3D hero: a slowly turning cluster of "ingredient molecules" (spheres joined by bonds) inside a ring of particles.
// `color` is the NOVA colour of the last result; the cluster eases toward it. `busy` speeds the spin while analysing.
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

const N = 22;

function fibonacciSphere(n: number, r: number) {
  const pts: THREE.Vector3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2;
    const rad = Math.sqrt(1 - y * y);
    const th = golden * i;
    const jitter = 0.75 + ((i * 37) % 10) / 22;
    pts.push(new THREE.Vector3(Math.cos(th) * rad, y, Math.sin(th) * rad).multiplyScalar(r * jitter));
  }
  return pts;
}

function Molecule({ color, busy }: { color: string; busy: boolean }) {
  const group = useRef<THREE.Group>(null);
  const target = useMemo(() => new THREE.Color(color), [color]);
  // one material + two geometries shared by every atom, so one lerp recolours the whole cluster
  const atomMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: "#3ddc97", emissive: "#3ddc97", emissiveIntensity: 0.35, roughness: 0.25, metalness: 0.4 }),
    [],
  );
  const bondMat = useMemo(() => new THREE.LineBasicMaterial({ color: "#3ddc97", transparent: true, opacity: 0.35 }), []);
  const big = useMemo(() => new THREE.SphereGeometry(0.2, 32, 32), []);
  const small = useMemo(() => new THREE.SphereGeometry(0.12, 24, 24), []);

  const atoms = useMemo(() => fibonacciSphere(N, 1.55), []);
  const bonds = useMemo(() => {
    const pos: number[] = [];
    atoms.forEach((a, i) =>
      atoms.forEach((b, j) => {
        if (j > i && a.distanceTo(b) < 1.05) pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
      }),
    );
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, [atoms]);

  useFrame((state, dt) => {
    if (!group.current) return;
    const speed = busy ? 1.6 : 0.18;
    group.current.rotation.y += dt * speed;
    group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.35) * 0.25;
    group.current.position.y = Math.sin(state.clock.elapsedTime * 0.8) * 0.08;
    const pulse = busy ? 1 + Math.sin(state.clock.elapsedTime * 8) * 0.04 : 1;
    group.current.scale.setScalar(pulse);
    atomMat.color.lerp(target, 0.04);
    atomMat.emissive.lerp(target, 0.04);
    bondMat.color.lerp(target, 0.04);
  });

  return (
    <group ref={group}>
      {atoms.map((p, i) => (
        <mesh key={i} position={p} geometry={i % 5 === 0 ? big : small} material={atomMat} />
      ))}
      <lineSegments geometry={bonds} material={bondMat} />
    </group>
  );
}

// mulberry32: tiny seeded PRNG returning floats in [0, 1)
function seededRandom(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function Particles() {
  const ref = useRef<THREE.Points>(null);
  const geom = useMemo(() => {
    const rand = seededRandom(7);                 // deterministic, so render stays pure
    const pos = new Float32Array(900 * 3);
    for (let i = 0; i < 900; i++) {
      const r = 2.6 + rand() * 2.2;
      const th = rand() * Math.PI * 2;
      const ph = Math.acos(2 * rand() - 1);
      pos.set([r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph) * 0.6, r * Math.sin(ph) * Math.sin(th)], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    return g;
  }, []);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y -= dt * 0.03;
  });
  return (
    <points ref={ref} geometry={geom}>
      <pointsMaterial size={0.018} color="#9fe8c8" transparent opacity={0.55} sizeAttenuation />
    </points>
  );
}

export default function Scene3D({ color = "#3ddc97", busy = false }: { color?: string; busy?: boolean }) {
  return (
    <Canvas camera={{ position: [0, 0, 5.2], fov: 45 }} dpr={[1, 1.75]} gl={{ antialias: true, alpha: true }}>
      <ambientLight intensity={0.35} />
      <pointLight position={[4, 4, 4]} intensity={40} color="#ffffff" />
      <pointLight position={[-4, -2, -3]} intensity={18} color="#7cc4ff" />
      <Molecule color={color} busy={busy} />
      <Particles />
    </Canvas>
  );
}
