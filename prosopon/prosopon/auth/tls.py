"""Dynamic local TLS certificate generator for mobile browser Secure Context."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate_self_signed_cert(
    output_dir: Path,
    common_name: str = "metis.local",
    valid_days: int = 365,
) -> Tuple[Path, Path]:
    """
    Generate an ephemeral self-signed RSA TLS certificate and private key.
    Enables W3C Secure Context over LAN so Android browsers permit microphone access.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cert_path = output_dir / "prosopon.crt"
    key_path = output_dir / "prosopon.key"

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    # Generate 2048-bit RSA private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Metis Local Stack"),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=valid_days))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("metis.local"),
                x509.IPAddress(ipaddress_from_str("127.0.0.1")),
                x509.IPAddress(ipaddress_from_str("0.0.0.0")),
            ]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    # Write private key
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


def ipaddress_from_str(ip_str: str):
    """Helper to convert IP string to ipaddress object."""
    import ipaddress
    return ipaddress.ip_address(ip_str)
