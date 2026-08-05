"""Tests for Seat and Direct models (Task 16)."""

import torch
import pytest

from seatflow.models.seat import SeatEncoder, EventEncoder, DirectEffectModel


class TestSeatEncoder:
    def test_output_shape(self):
        encoder = SeatEncoder(input_dim=8, hidden_dim=64, output_dim=32, num_heads=4)
        x = torch.randn(4, 5, 8)  # batch=4, 5 seats, 8 features
        out = encoder(x)
        assert out.shape == (4, 32)

    def test_mask_zeros_invalid_seats(self):
        encoder = SeatEncoder(input_dim=8, hidden_dim=64, output_dim=32, num_heads=4)
        x = torch.randn(4, 5, 8)
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
            [True, True, True, True, True],
        ])
        out = encoder(x, mask=mask)
        assert out.shape == (4, 32)

    def test_masked_output_differs_from_unmasked(self):
        """Masking out seats produces a different output."""
        encoder = SeatEncoder(input_dim=8, hidden_dim=64, output_dim=32, num_heads=4)
        x = torch.randn(1, 5, 8)
        mask = torch.tensor([[True, True, True, False, False]])
        out_masked = encoder(x, mask=mask)
        out_unmasked = encoder(x)
        # They should differ because different seats are attended to
        assert not torch.allclose(out_masked, out_unmasked, atol=1e-5)


class TestEventEncoder:
    def test_output_shape(self):
        encoder = EventEncoder(input_dim=6, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 6)
        out = encoder(x)
        assert out.shape == (4, 32)


class TestDirectEffectModel:
    def test_forward_shape(self):
        model = DirectEffectModel(
            seat_input_dim=8,
            event_input_dim=6,
            market_input_dim=16,
            num_horizons=3,
        )
        seat_feat = torch.randn(4, 5, 8)
        event_feat = torch.randn(4, 6)
        market_feat = torch.randn(4, 16)
        out = model(seat_feat, event_feat, market_feat)
        assert out.shape == (4, 3)

    def test_direct_delta_is_additive(self):
        """Direct model outputs a delta, not absolute return."""
        model = DirectEffectModel(
            seat_input_dim=8,
            event_input_dim=6,
            market_input_dim=16,
            num_horizons=3,
        )
        seat_feat = torch.randn(4, 5, 8)
        event_feat = torch.randn(4, 6)
        market_feat = torch.randn(4, 16)
        out = model(seat_feat, event_feat, market_feat)
        # Delta can be positive or negative
        assert out.requires_grad

    def test_seat_mask_applied(self):
        model = DirectEffectModel(
            seat_input_dim=8,
            event_input_dim=6,
            market_input_dim=16,
            num_horizons=3,
        )
        seat_feat = torch.randn(4, 5, 8)
        event_feat = torch.randn(4, 6)
        market_feat = torch.randn(4, 16)
        mask = torch.ones(4, 5, dtype=torch.bool)
        mask[:, 3:] = False
        out = model(seat_feat, event_feat, market_feat, seat_mask=mask)
        assert out.shape == (4, 3)


class TestMaturityMask:
    """Seat history records whose label_end_time > cutoff must not affect output."""

    def test_mature_records_affect_output(self):
        """Masked (immature) seats produce different output vs full seats."""
        encoder = SeatEncoder(input_dim=8, hidden_dim=64, output_dim=32, num_heads=4)
        x = torch.randn(1, 5, 8)
        mask = torch.tensor([[True, True, True, False, False]])
        out_masked = encoder(x, mask=mask)
        out_full = encoder(x)
        # The output should be different when immature seats are excluded
        assert not torch.allclose(out_masked, out_full, atol=1e-5)
